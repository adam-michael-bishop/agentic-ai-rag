import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GOOGLE_GEMINI_API_KEY = os.environ["GOOGLE_GEMINI_API_KEY"]

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_GEMINI_API_KEY
)