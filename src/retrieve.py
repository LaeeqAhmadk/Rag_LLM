import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from custom_logger import logger
from exception import CustomException
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def format_docs(docs):
    """
    Format the documents retrieved by Qdrant to a string suitable for passing to the LLM.
    
    Args:
        docs (list): The retrieved documents to format.
    
    Returns:
        str: The formatted documents as a string.
    """
    return "\n".join([doc.page_content for doc in docs])

def retrieve_answer_from_docs(question: str):
    """
    Retrieve the answer to a question from the documents.

    Args:
        question (str): The question to answer.

    Returns:
        str: The generated answer.
    """
    try:
        # Environment variables
        qdrant_url = os.getenv('qdrant_url')
        qdrant_api_key = os.getenv('qdrant_api')
        groq_api_key = os.getenv('groq_api')

        if not all([qdrant_url, qdrant_api_key, groq_api_key]):
            raise ValueError("One or more environment variables are missing.")

        # Define the prompt template for LLM interaction
        prompt = PromptTemplate(
            template="""# Your role
                        You are an expert at understanding the intent of the questioner and providing optimal answers from the documents.

                        # Instruction
                        Your task is to answer the question using the following retrieved context delimited by XML tags.

                        <retrieved context>
                        Retrieved Context:
                        {context}
                        </retrieved context>

                        # Question:
                        {question}""",
            input_variables=["context", "question"]
        )

        # Load the sentence transformer model
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

        # Initialize Qdrant client and Qdrant vector store
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        qdrant = Qdrant(client=qdrant_client, collection_name="rag", embeddings=embeddings_model)
        retriever = qdrant.as_retriever(search_kwargs={"k": 20})

        # Set up the ChatGroq LLM for answering
        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0, max_tokens=None, timeout=None, max_retries=2)

        # Chain the retriever, formatter, and LLM together
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain and get the answer
        answer = rag_chain.invoke(question)
        logger.info("Answer retrieved successfully")
        return answer

    except ValueError as ve:
        logger.error("Environment variable error: %s", ve)
        raise CustomException(ve, sys)
    except Exception as e:
        logger.error("Error retrieving answer: %s", str(e))
        raise CustomException(e, sys)
