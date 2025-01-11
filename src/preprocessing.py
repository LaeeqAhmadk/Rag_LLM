import os
import sys
sys.path.append("D:/Projects/Rag_LLM/Rag_LLM")
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from custom_logger import logger
from exception import CustomException

def load_documents(file_path: str):
    """
    Load documents from a PDF file.
    
    Args:
        file_path(str): Path to the PDF file.
        
    Returns:
        list: List of documents.
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info("Documents loaded successfully from %s", file_path)
        return documents
    except Exception as e:
        raise CustomException(e, sys)
    
def split_documents(documents: list, chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Split documents into chunks.

    Args:
        documents(list): List of documents.
        chunk_size(int): Size of each chunk.
        chunk_overlap(int): Overlap between chunks.

    Returns:
        list: List of chunks.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        chunks = text_splitter.split_documents(documents)
        logger.info("Documents split into chunks successfully")
        return texts
    except Exception as e:
        raise CustomException(e, sys)
    