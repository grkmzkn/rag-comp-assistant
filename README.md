# Corporate Knowledge Assistant

This project implements a RAG (Retrieval Augmented Generation) based chatbot that helps users access and query department-specific information from PDF documents. The system uses FAISS for efficient similarity search and Google's Gemini Pro model for generating responses.

> **Note**: This is a demonstration project using synthetic data. The included dataset is artificially generated to showcase the capabilities of the RAG system and does not contain any real company information.

## Features

- 📚 PDF document processing and indexing
- 🔍 Semantic search using FAISS
- 💡 Intelligent responses using Google Gemini
- 💬 Interactive chat interface (CLI and Streamlit)
- 🎯 Context-aware answers
- 🔄 Overlapping text chunks for better context preservation

## Requirements

- Python 3.10 or higher
- Required packages (listed in requirements.txt)
- Google Gemini API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/grkmzkn/rag-comp-assistant.git
cd rag-comp-assistant
```

2. Create a virtual environment:
```bash
python -m venv env
```

3. Activate the virtual environment:
- Windows:
```bash
.\env\Scripts\activate
```
- Unix/MacOS:
```bash
source env/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
- Create a `.env` file in the root directory
- Add your Google Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Usage

### About the Dataset

This project uses a synthetic dataset created for demonstration purposes. The data:
- Simulates internal department documentation
- Contains artificial but realistic business scenarios
- Is structured to demonstrate RAG capabilities
- Does not contain any real company information

### Preparing Documents

1. Place your PDF documents in the `data` directory
2. Each PDF should contain department-specific information
3. Recommended document structure:
   - Clear topic sections
   - 75-100 words per topic
   - Multiple topics per department
   - Similar to the synthetic dataset structure

### Running the Chatbot

#### CLI Version
```bash
python src/chatbot.py
```

#### Streamlit Interface
```bash
streamlit run src/chatbot_ui.py
```

## System Architecture

The system uses a RAG architecture with the following components:

1. **Document Processing**:
   - PDF text extraction
   - Text chunking with overlap
   - Vector embeddings generation

2. **Search System**:
   - FAISS vector database
   - Cosine similarity search
   - Top-k retrieval

3. **Response Generation**:
   - Context retrieval
   - Prompt engineering
   - Gemini model integration

## Configuration

Key parameters in the system:
- Chunk size: 150 words
- Chunk overlap: 50 words
- Retrieved documents: 5
- Temperature: 0.5
- Maximum output tokens: 2048

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Generative AI for Gemini model
- Facebook Research for FAISS
- Sentence Transformers for embeddings
- Streamlit for the UI framework