import os
from dotenv import load_dotenv
from src.preprocessing import load_documents, split_documents
from src.index import store_documents_to_qdrant
from src.retrieve import retrieve_answer_from_docs
from src.utils import format_docs
from custom_logger import logger
from exception import CustomException

# Load environment variables from the .env file
load_dotenv()

def retriever(question: str):
    """
    Retrieve an answer to a question from the indexed documents.

    Args:
        question (str): The input question for retrieval.

    Returns:
        str: The retrieved answer.
    """
    try:
        logger.info("Starting retrieval process...")

        # --- Step 1: Load and preprocess documents (Uncomment if needed) ---
        # file_path = os.getenv('DOCUMENTS_PATH')  # Set the path to your documents in the .env file
        # if not file_path:
        #     raise ValueError("DOCUMENTS_PATH environment variable is not set.")
        # 
        # logger.info(f"Loading documents from {file_path}...")
        # documents = load_documents(file_path)
        # texts = split_documents(documents)

        # --- Step 2: Store documents in Qdrant (Uncomment if needed) ---
        # logger.info("Storing documents in Qdrant...")
        # qdrant = store_documents_to_qdrant(texts)

        # --- Step 3: Retrieve the answer ---
        logger.info(f"Retrieving answer for the question: {question}")
        answer = retrieve_answer_from_docs(question)

        logger.info(f"Answer retrieved: {answer}")
        return answer

    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error(f"An error occurred during the retrieval process: {e}")
        raise CustomException(e, sys)

# Example usage (For testing purposes)
if __name__ == "__main__":
    try:
        # Example question
        sample_question = "What are the documents uploaded in the database?"
        response = retriever(sample_question)
        print(f"Question: {sample_question}\nAnswer: {response}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
