import os
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.services.milvus import vectorstore
from langchain_google_genai import GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt for RAG
prompt = hub.pull("rlm/rag-prompt")

# Initialize the Gemini LLM
GOOGLE_GEMINI_API_KEY = os.environ["GOOGLE_GEMINI_API_KEY"]

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_GEMINI_API_KEY
)

def invoke_with_context(question: str):
    qa_chain = (
        {
            "context": vectorstore.as_retriever() | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain.invoke(question)

def split_text(contents: str, chunk_size=1000, chunk_overlap=100) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(contents)