from langchain_milvus import Milvus
from app.services.embeddings import embeddings

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

def add_documents_to_vectorstore(docs):
    """
    Adds a list of Document objects to the Milvus vector store.
    """
    vectorstore.add_documents(docs)