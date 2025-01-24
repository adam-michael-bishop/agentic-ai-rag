from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.services.langchain import invoke_with_context, split_text
from app.services.milvus import add_documents_to_vectorstore
from langchain.docstore.document import Document

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a document file.
    The file content is split into chunks, embedded, and stored in Milvus.
    """
    try:
        # Read and decode the uploaded file
        contents = await file.read()
        contents = contents.decode('utf-8')  # Adjust decoding as needed
        texts = split_text(contents)

        # Create Document objects
        docs = [Document(page_content=text) for text in texts]

        # Add documents to the vector store
        add_documents_to_vectorstore(docs)

        return {"message": "File uploaded and processed successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@router.post("/query/")
async def process_query(request: QueryRequest):
    """
    Endpoint to handle a user query.
    Uses a QA chain to find answers based on the uploaded documents.
    """
    try:
        # Generate the answer using the QA chain
        answer = invoke_with_context(request.question)

        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})