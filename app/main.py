from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os

# LangChain and Milvus imports
from langchain_milvus import Milvus
from langchain import hub
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough

# Import the Gemini LLM
# import google.generativeai as genai

# Initialize the FastAPI app
app = FastAPI()

# Prompt for RAG
prompt = hub.pull("rlm/rag-prompt")

# Initialize the Gemini LLM
GOOGLE_GEMINI_API_KEY = os.environ["GOOGLE_GEMINI_API_KEY"]
llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_GEMINI_API_KEY)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_GEMINI_API_KEY)

# Milvus configuration
milvus_collection_name = "documents_collection"
milvus_connection_args = {"host": "localhost", "port": "19530"}

# Initialize Milvus vector store
vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name=milvus_collection_name,
    connection_args=milvus_connection_args,
    auto_id=True,
)

# Define a Pydantic model for the query request
class QueryRequest(BaseModel):
    question: str

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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

@app.post("/query/")
async def process_query(request: QueryRequest):
    """
    Endpoint to handle a user query.
    Uses a QA chain to find answers based on the uploaded documents.
    """
    try:
        # Set up the QA chain
        qa_chain = (
            {
                "context": vectorstore.as_retriever() | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate the answer
        answer = qa_chain.invoke(request.question)
        
        return {"answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)