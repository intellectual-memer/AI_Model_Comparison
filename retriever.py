import streamlit as st
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile
import os
from langchain.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

def get_retriever(uploaded_files):
    """
    Dynamically processes uploaded files (txt, pdf, docx), loads their content,
    and creates a retriever using FAISS.
    """
    temp_dir = tempfile.mkdtemp()
    documents = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load the file based on its type
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            continue
        
        documents.extend(loader.load())

    # Create FAISS index

    # embeddings = OpenAIEmbeddings(openai_api_key="your_openai_api_key")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore.as_retriever()
