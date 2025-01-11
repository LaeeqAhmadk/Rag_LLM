from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # Ensures input validation
from src.main import retriever  # Import retriever from your main logic module

app = FastAPI()  # Initialize FastAPI app


class QueryRequest(BaseModel):
    """Schema for query request validation."""
    question: str


@app.post("/ask/")
async def ask_question(query: QueryRequest):
    """
    Endpoint to handle user queries.

    Args:
        query (QueryRequest): The input query containing a question string.

    Returns:
        dict: A response containing the input question and the generated answer.
    """
    # Validate the input
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Call the retriever function for the answer
        answer = retriever(query.question)
    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(
            status_code=500, detail=f"An error occurred while processing the query: {str(e)}"
        )

    return {"question": query.question, "answer": answer}


if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
