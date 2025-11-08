import os
import json
import time
import PyPDF2
import faiss
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize sentence transformer model from local directory
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'multi-qa-MiniLM-L6-cos-v1') # from local folder
# embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1') # from cache
embedding_model = SentenceTransformer(model_path)

def read_pdf(pdf_path):
    """Reads a PDF file and extracts its text content.
    
    Args:
        pdf_path (str): The path to the PDF file to be read.
        
    Returns:
        str: The extracted text content from all pages of the PDF.
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def calculate_dynamic_chunk_size(text):
    """Calculate dynamic chunk size based on document length.
    
    Args:
        text (str): The input text.
    
    Returns:
        tuple: (chunk_size, overlap_size) in words
    """
    total_words = len(word_tokenize(text))
    
    if total_words < 500:
        return 100, 30  # Small documents
    elif total_words < 1000:
        return 150, 50  # Medium documents
    else:
        return 200, 70  # Large documents

def split_into_chunks(text, chunk_size=150, overlap=50):
    """Splits text into smaller chunks using sentence-level tokenization and dynamic sizing.
    
    This function:
    1. Splits text into sentences using NLTK
    2. Calculates dynamic chunk size based on document length
    3. Groups sentences while respecting word limits
    4. Ensures sentence integrity is maintained
    5. Provides overlap between chunks for context preservation
    
    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): Target size in words. May be adjusted dynamically.
        overlap (int, optional): Target overlap in words. May be adjusted dynamically.
        
    Returns:
        list[str]: A list of text chunks, each containing complete sentences
                  and approximately following the target size.
    """
    # Calculate dynamic chunk size based on document length
    dynamic_chunk_size, dynamic_overlap = calculate_dynamic_chunk_size(text)
    chunk_size = dynamic_chunk_size
    overlap = dynamic_overlap
    
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # If text is very short, return as single chunk
    if len(sentences) <= 3:
        return [text]
    
    chunks = []
    current_chunk = []
    current_size = 0
    last_chunk_size = 0
    
    for sentence in sentences:
        sentence_words = word_tokenize(sentence)
        sentence_size = len(sentence_words)
        
        # If a single sentence is very long, split it
        if sentence_size > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split long sentence while keeping words together
            words = sentence.split()
            for i in range(0, len(words), chunk_size):
                chunk = words[i:i + chunk_size]
                chunks.append(" ".join(chunk))
            continue
        
        # Check if adding this sentence would exceed chunk size
        if current_size + sentence_size > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Calculate overlap
            overlap_start = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_start:]
            current_size = sum(len(word_tokenize(s)) for s in current_chunk)
            
        current_chunk.append(sentence)
        current_size += sentence_size
        
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embeddings(text):
    """Generates normalized embeddings for the input text using sentence-transformers model.
    
    Args:
        text (str): The input text to generate embeddings for.
        
    Returns:
        numpy.ndarray: The normalized embedding vector for the input text.
    """
    # Normalize the embeddings for cosine similarity
    embedding = embedding_model.encode(text)
    normalized_embedding = embedding / np.linalg.norm(embedding)
    return normalized_embedding

class RAGSystem:
    def __init__(self, dimension=384, index_path="models/faiss_index"):  # sentence-transformers embedding dimension
        self.index_path = index_path
        self.texts_path = f"{index_path}_texts.npy"
        self.state_path = f"{index_path}_state.json"
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
        if self._should_update_index():
            # Create new index if state changed
            print("Changes detected in data directory. Updating index...")
            self.index = faiss.IndexFlatIP(dimension)
            self.texts = []
            self._process_all_documents()
        else:
            # Load existing index
            print("Loading existing index...")
            self.index = faiss.read_index(self.index_path)
            self.texts = list(np.load(self.texts_path, allow_pickle=True))
    
    def _get_data_state(self):
        """Get current state of data directory including file count and latest modification time."""
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
        latest_mod_time = max(os.path.getmtime(os.path.join(self.data_dir, f)) 
                            for f in files) if files else 0
        return {
            'file_count': len(files),
            'latest_modification': latest_mod_time
        }
    
    def _should_update_index(self):
        """
        Check if index needs to be updated based on data directory state.
        Returns True if:
        - State file doesn't exist
        - Index files don't exist
        - Number of PDF files changed
        - Any PDF file was modified
        """
        current_state = self._get_data_state()
        
        # If no state file exists or no index files exist, update is needed
        if not os.path.exists(self.state_path) or \
           not os.path.exists(self.index_path) or \
           not os.path.exists(self.texts_path):
            self._save_state(current_state)
            return True
            
        # Load previous state
        with open(self.state_path, 'r') as f:
            previous_state = json.load(f)
            
        # Check if state changed
        if previous_state['file_count'] != current_state['file_count'] or \
           previous_state['latest_modification'] != current_state['latest_modification']:
            self._save_state(current_state)
            return True
            
        return False
    
    def _save_state(self, state):
        """Save current state (file count and modification time) to JSON file."""
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w') as f:
            json.dump(state, f)
            
    def save_index(self):
        """Saves the FAISS index and text chunks to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.texts_path, np.array(self.texts, dtype=object))
        
    def add_document(self, pdf_path):
        """Processes a PDF document and adds its content to the RAG system.
        
        This method reads the PDF, splits it into chunks, generates embeddings,
        and stores both the embeddings and original text chunks.
        
        Args:
            pdf_path (str): The path to the PDF document to be processed.
        """
        text = read_pdf(pdf_path)
        chunks = split_into_chunks(text)
        
        for chunk in chunks:
            embedding = get_embeddings(chunk)
            self.index.add(np.array([embedding], dtype=np.float32))
            self.texts.append(chunk)
    
    def _process_all_documents(self):
        """
        Process all PDF documents in data directory and save the index.
        This method:
        1. Reads each PDF file in the data directory
        2. Processes them into chunks
        3. Creates embeddings and adds to FAISS index
        4. Saves the final index and state
        """
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.data_dir, filename)
                print(f"Processing file: {filename}")
                self.add_document(pdf_path)
        
        # Save index and state after processing all documents
        self.save_index()
        current_state = self._get_data_state()
        self._save_state(current_state)
    
    def search(self, query, k=5):
        """Searches for the k most relevant text chunks for a given query.
        
        Args:
            query (str): The search query text.
            k (int, optional): The number of most similar chunks to retrieve. Defaults to 5.
            
        Returns:
            list[str]: The k most relevant text chunks based on embedding similarity.
        """
        query_embedding = get_embeddings(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]

    def get_response(self, query, chat_session=None, max_retries=3):
        """Generates a response using Gemini model based on relevant context.
        
        This method:
        1. Retrieves relevant text chunks using similarity search
        2. Creates a prompt with the context and query
        3. Generates a response using the Gemini model with retry mechanism
        
        Args:
            query (str): The user's question or query.
            chat_session: Optional chat session to maintain conversation context.
            max_retries: Maximum number of retry attempts for API calls.
            
        Returns:
            str: The generated response from the Gemini model.
            
        Raises:
            Exception: If all retry attempts fail
        """
        relevant_chunks = self.search(query)
        context = "\n".join(relevant_chunks)

        # Model configuration
        generation_config = {
            "temperature": 0.5,
            "max_output_tokens": 2048,
            "top_p": 0.8,
            "top_k": 40,
        }

        system_prompt = """Sen şirket içi dokümanlarda bulunan bilgileri kullanarak yanıt veren bir asistansın.
        Bu kurallara uymalısın:
        1. Sadece verilen bağlam içindeki bilgileri kullan
        2. Verilen bağlamı dikkatlice analiz et ve doğrudan veya dolaylı olarak ilgili bilgiyi ara
        3. Farklı kelimelerle sorulmuş olsa bile, aynı konuyla ilgili bilgileri tanı ve kullan
           Örneğin:
           - "Şifre değiştirme sıklığı" = "Şifre yenileme kuralı"
           - "Kaç günde bir" = "Ne sıklıkla" = "Hangi aralıklarla"
        4. Eğer bağlamda kesinlikle ilgili bir bilgi yoksa şu şekilde yanıt ver:
           "Bu konuda bilgim bulunmuyor.
           
           Sorunuzun aşağıdaki departmanlardan biriyle ilgili olduğunu düşünüyorsanız, ilgili departmana mail iletebilirsiniz:
           
           - İnsan Kaynakları: ik@sirket.com
           - Bilgi Teknolojileri: it@sirket.com
           - Finans ve Muhasebe: finans@sirket.com
           - Satış ve Pazarlama: satis@sirket.com
           - İdari İşler: idari@sirket.com
           - Üretim: uretim@sirket.com
           - Kalite Kontrol: kalite@sirket.com
           - Ar-Ge: arge@sirket.com
           - Hukuk: hukuk@sirket.com"
        5. Yanıtların kısa ve net olsun
        6. Profesyonel ve kibar bir dil kullan
        7. Tüm yanıtlarını Türkçe olarak ver
        8. Şüpheli durumlarda, bağlamı daha geniş yorumla ve ilgili olabilecek bilgileri değerlendir
        9. Önceki konuşma bağlamını dikkate al ve tutarlı cevaplar ver
        """

        prompt = f"""Aşağıda verilen bağlam bilgisini kullan:
        ---------------------
        {context}
        ---------------------
        Sadece yukarıdaki bağlam bilgisini kullanarak şu soruyu yanıtla:
        Soru: {query}
        Yanıt: """

        for attempt in range(max_retries):
            try:
                if chat_session is None:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    model.generation_config = generation_config
                    chat_session = model.start_chat(history=[])
                    chat_session.send_message(system_prompt)
                
                # Get response using existing chat session
                response = chat_session.send_message(prompt)
                return response.text
                
            except Exception as e:
                if attempt == max_retries - 1:  # Son deneme başarısız olduysa
                    raise Exception(f"Gemini API hatası ({max_retries} denemeden sonra): {str(e)}")
                print(f"API çağrısı başarısız (deneme {attempt + 1}/{max_retries}), yeniden deneniyor...")
                time.sleep(1)  # 1 saniye bekle ve tekrar dene