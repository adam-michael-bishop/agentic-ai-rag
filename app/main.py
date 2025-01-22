from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os

# LangChain and Milvus imports
from langchain.vectorstores import Milvus
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Import the Gemini LLM
import google.generativeai as genai

# Initialize the FastAPI app
app = FastAPI()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Milvus configuration
milvus_collection_name = "documents_collection"
milvus_connection_args = {"host": "localhost", "port": "19530"}

# Initialize Milvus vector store
vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name=milvus_collection_name,
    connection_args=milvus_connection_args,
)

# Initialize the Gemini LLM
GOOGLE_GEMINI_API_KEY = os.environ["GOOGLE_GEMINI_API_KEY"]
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
llm = genai.GenerativeModel('gemini-1.5-flash')

# Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    question: str

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a document file.
    The file content is split into chunks, embedded, and stored in Milvus.
    """
    try:
        # Read and decode the uploaded file
        contents = await file.read()
        contents = contents.decode('utf-8')  # Adjust decoding as needed
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(contents)
        
        # Create Document objects
        docs = [Document(page_content=text) for text in texts]
        
        # Add documents to the vector store
        vectorstore.add_documents(docs)
        
        return {"message": "File uploaded and processed successfully."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """
    Endpoint to handle user questions.
    Uses a RetrievalQA chain to find answers based on the uploaded documents.
    """
    try:
        # Retrieve relevant documents using the vector store
        retriever = vectorstore.as_retriever()
        
        # Set up the RetrievalQA chain with the LLM and retriever
        qa_chain = RetrievalQA(llm=llm, retriever=retriever)
        
        # Generate the answer
        answer = qa_chain.run(request.question)
        
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)