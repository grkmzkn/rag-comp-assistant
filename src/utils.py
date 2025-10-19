import os
import PyPDF2
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize sentence transformer model - using a lightweight model for memory efficiency
# embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller model, more RAM friendly

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

def split_into_chunks(text, chunk_size=150, overlap=50):
    """Splits text into smaller chunks with overlap for better context preservation.
    
    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The target size of each chunk in words. Defaults to 150.
        overlap (int, optional): The number of words to overlap between chunks. Defaults to 50.
        
    Returns:
        list[str]: A list of text chunks, each containing approximately chunk_size words
                  with overlap between consecutive chunks.
    """
    words = text.split()
    chunks = []
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(words) <= chunk_size:
        return [text]
    
    for i in range(0, len(words), chunk_size - overlap):
        # Get chunk_size words or remaining words if less
        chunk = words[i:i + chunk_size]
        # Only add if chunk is not empty and has meaningful length
        if chunk and len(chunk) > overlap:
            chunks.append(" ".join(chunk))
    
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
    def __init__(self, dimension=384):  # sentence-transformers embedding dimension
        # Use cosine similarity instead of L2 distance
        self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (for cosine similarity)
        self.texts = []
        
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

    def get_response(self, query):
        """Generates a response using Gemini model based on relevant context.
        
        This method:
        1. Retrieves relevant text chunks using similarity search
        2. Creates a prompt with the context and query
        3. Generates a response using the Gemini model
        
        Args:
            query (str): The user's question or query.
            
        Returns:
            str: The generated response from the Gemini model.
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
        """

        prompt = f"""Aşağıda verilen bağlam bilgisini kullan:
        ---------------------
        {context}
        ---------------------
        Sadece yukarıdaki bağlam bilgisini kullanarak şu soruyu yanıtla:
        Soru: {query}
        Yanıt: """

        model = genai.GenerativeModel('gemini-2.5-flash')
        model.generation_config = generation_config
        
        # Create chat session with system prompt
        chat = model.start_chat(history=[])
        chat.send_message(system_prompt)
        
        # Get response
        response = chat.send_message(prompt)
        return response.text