import sys
from custom_logger import logger
from exception import CustomException

def format_docs(docs):
    """
    Format documents for retrieval output.

    Args:
        docs (list): List of document objects.

    Returns:
        str: Formatted documents as a single string.
    """
    try:
        if not docs:
            raise ValueError("No documents provided for formatting.")
        
        formatted_docs = []
        for doc in docs:
            # Format the metadata into a string (if metadata exists)
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata_str = ', '.join(f"{key}: {value}" for key, value in doc.metadata.items())
                doc_str = f"{doc.page_content}\nMetadata: {metadata_str}"
            else:
                # Handle case where no metadata exists
                doc_str = f"{doc.page_content}\nMetadata: None"
            
            # Append to the list of formatted documents
            formatted_docs.append(doc_str)

        # Join all formatted documents with double newlines
        formatted_string = "\n\n".join(formatted_docs)
        
        logger.info("Documents formatted successfully")
        return formatted_string

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error(f"Error formatting documents: {e}")
        raise CustomException(e, sys)
