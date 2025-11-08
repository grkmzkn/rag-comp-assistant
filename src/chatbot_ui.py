"""Streamlit UI for Department Information Chatbot.

This module provides a web-based user interface for the RAG chatbot using Streamlit.
It includes:
- Chat history management
- Clean and responsive UI
- Session state handling
- PDF file path configuration
"""

import streamlit as st
import google.generativeai as genai
from utils import RAGSystem
import os

def initialize_session_state():
    """Initializes Streamlit session state variables.
    
    This function sets up persistent storage for:
    - Chat history
    - RAG system instance
    - Chat session for maintaining conversation context
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
        
    if "chat_session" not in st.session_state:
        model = genai.GenerativeModel('gemini-2.5-flash')
        st.session_state.chat_session = model.start_chat(history=[])

def initialize_rag_system():
    """Initializes the RAG system with documents from the data directory.
    
    Returns:
        RAGSystem: An initialized RAG system containing processed documents.
    """
    # RAGSystem sınıfı otomatik olarak dosya değişikliklerini kontrol eder
    # ve sadece gerektiğinde indeksi günceller
    with st.spinner("PDF dosyaları kontrol ediliyor..."):
        rag = RAGSystem()
        
    return rag

def main():
    """Main function that sets up and runs the Streamlit interface."""
    
    # Page configuration
    st.set_page_config(
        page_title="Kurumsal Bilgi Asistanı",
        page_icon="💬",
        layout="wide"
    )
    
    # Title
    st.title("Kurumsal Bilgi Asistanı 💬")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG system for first time
    if st.session_state.rag_system is None:
        with st.spinner("Sistem başlatılıyor... PDF dosyaları kontrol ediliyor..."):
            st.session_state.rag_system = initialize_rag_system()
        st.success("Sistem hazır! PDF dosyaları yüklendi ve indeks güncel.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Sorunuzu yazın..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                try:
                    response = st.session_state.rag_system.get_response(
                        prompt, 
                        chat_session=st.session_state.chat_session
                    )
                    st.markdown(response)
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"Üzgünüm, bir hata oluştu: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Clear chat button
    if st.sidebar.button("Sohbeti Temizle"):
        st.session_state.messages = []
        # Reset chat session
        model = genai.GenerativeModel('gemini-2.5-flash')
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()

if __name__ == "__main__":
    main()