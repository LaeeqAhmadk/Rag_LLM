import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from custom_logger import logger
from exception import CustomException

def load_documents(file_path: str):
    """
    Load documents from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        list: List of documents.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")

        logger.info("Loading documents from: %s", file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info("Documents loaded successfully from %s", file_path)
        return documents

    except FileNotFoundError as fe:
        logger.error("File not found: %s", file_path)
        raise CustomException(fe, sys)
    except Exception as e:
        logger.error("Error while loading documents: %s", str(e))
        raise CustomException(e, sys)

def split_documents(documents: list, chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Split documents into smaller chunks.

    Args:
        documents (list): List of documents to be split.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        list: List of text chunks.
    """
    try:
        if not documents or len(documents) == 0:
            raise ValueError("No documents provided to split.")

        logger.info("Splitting documents into chunks with size %d and overlap %d", chunk_size, chunk_overlap)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        logger.info("Documents split into %d chunks successfully", len(chunks))
        return chunks

    except ValueError as ve:
        logger.error("Invalid input for splitting documents: %s", ve)
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error("Error while splitting documents: %s", str(e))
        raise CustomException(e, sys)
