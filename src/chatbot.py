"""Department Information Chatbot using RAG (Retrieval Augmented Generation).

This module implements a chatbot interface that uses RAG architecture to provide
accurate responses to queries about different departments. It uses:
- PDF documents as the knowledge base
- FAISS for efficient similarity search
- Gemini model for response generation
- Sentence transformers for text embeddings

The system processes PDF documents from the data directory and uses their content
to answer user queries in a conversational manner.
"""

from utils import RAGSystem
import os

def initialize_rag_system():
    """Initializes and populates the RAG system with documents from the data directory.
    
    This function:
    1. Creates a new RAGSystem instance
    2. Scans the data directory for PDF files
    3. Processes each PDF and adds it to the system
    
    Returns:
        RAGSystem: An initialized RAG system containing processed documents.
    """
    rag = RAGSystem()
    
    pdf_dir = "./data"
    
    # Process all PDF files in the directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            rag.add_document(pdf_path)
    
    return rag

def main():
    """Main function that runs the chatbot interface.
    
    This function:
    1. Initializes the RAG system
    2. Starts an interactive loop for user queries
    3. Processes each query and displays responses
    4. Continues until the user types 'quit'
    
    The chatbot provides department-specific information based on
    the content of PDF documents in the data directory.
    """
    print("Kurumsal Bilgi Asistanına Hoş Geldiniz")
    print("Çıkmak için 'quit' yazın.")
    
    # Initialize RAG system
    rag = initialize_rag_system()
    
    while True:
        # Get user query
        query = input("\nSorunuz: ").strip()
        
        if query.lower() == 'quit':
            break
        
        try:
            # Generate response
            response = rag.get_response(query)
            print("\nCevap:", response)
        except Exception as e:
            print(f"\nHata oluştu: {str(e)}")

if __name__ == "__main__":
    main()