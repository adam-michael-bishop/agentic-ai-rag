# **Agentic AI RAG App**

A FastAPI application that implements Retrieval-Augmented Generation (RAG) using LangChain, Google Generative AI (Gemini), and Milvus for vector storage. The app allows users to upload documents, processes them to create embeddings, stores them in Milvus, and provides a query endpoint to answer user questions based on the uploaded documents.

## **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [API Endpoints](#api-endpoints)
  - [Upload File Endpoint](#1-upload-file-endpoint-upload_file)
  - [Query Endpoint](#2-query-endpoint-query)
- [Testing the Endpoints](#testing-the-endpoints)
- [Project Components](#project-components)
  - [Main Application](#main-application-appmainpy)
  - [API Routes](#api-routes-appapiroutespy)
  - [Services](#services-appservices)

## **Overview**

This application demonstrates how to build a Retrieval-Augmented Generation (RAG) system using:

- **FastAPI**: A modern, fast web framework for building APIs with Python.
- **LangChain**: A framework for building applications with large language models (LLMs).
- **Google Generative AI (Gemini)**: An advanced LLM provided by Google.
- **Milvus**: An open-source vector database for storing and retrieving high-dimensional vectors.

The app allows users to:

1. **Upload Documents**: Users can upload text files containing documents.
2. **Process and Store**: The app splits documents into chunks, generates embeddings using Google Generative AI Embeddings, and stores them in Milvus.
3. **Query**: Users can ask questions related to the uploaded documents, and the app uses the QA chain to generate answers based on the stored information.

## **Features**

- **Document Uploading**: Upload text files to the application.
- **Text Chunking**: Split documents into manageable chunks for processing.
- **Embedding Generation**: Generate embeddings using Google Generative AI Embeddings.
- **Vector Storage**: Store embeddings in Milvus for efficient retrieval.
- **Question Answering**: Ask questions and receive answers based on the uploaded documents.
- **Modular Code Structure**: Clean separation of concerns using services and API routes.

## **API Endpoints**

### **1. Upload File Endpoint (`/upload_file/`)**

- **Method**: `POST`
- **Description**: Upload a text file to the server. The content is split, embedded, and stored in Milvus.
- **Form Data**:
  - `file`: The text file to upload.

### **2. Query Endpoint (`/query/`)**

- **Method**: `POST`
- **Description**: Submit a question to the server. The app uses the QA chain to generate an answer based on the uploaded documents.
- **Request Body** (JSON):

  ```json
  {
    "question": "Your question here"
  }
  ```

## **Testing the Endpoints**

### **Using `curl`**

**Upload File**:

```bash
curl -X POST "http://0.0.0.0:8000/upload_file/" -F "file=@your_document.txt"
```

**Query**:

```bash
curl -X POST "http://0.0.0.0:8000/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

### **Using Postman**

1. **Upload File**:
   - Set method to `POST` and URL to `http://0.0.0.0:8000/upload_file/`.
   - In the `Body` tab, select `form-data`.
   - Add a key `file`, set type to `File`, and select your file.
   - Send the request.

2. **Query**:
   - Set method to `POST` and URL to `http://0.0.0.0:8000/query/`.
   - In the `Body` tab, select `raw`, and choose `JSON` format.
   - Enter the JSON body:

     ```json
     {
       "question": "Your question here"
     }
     ```

   - Send the request.

## **Project Components**

### **Main Application: `app/main.py`**

- Initializes the FastAPI app.
- Includes the API router from `app/api/routes.py`.

### **API Routes: `app/api/routes.py`**

Defines the API endpoints for:

- **`/upload_file/`**: Handles file uploads.
- **`/query/`**: Processes user queries.

Key functions:

- **`upload_file`**:
  - Reads and decodes the uploaded file.
  - Splits the content into chunks.
  - Creates `Document` objects.
  - Adds documents to the vector store.

- **`process_query`**:
  - Invokes the QA chain with the user's question.
  - Returns the generated answer.

### **Services: `app/services/`**

Contains service modules that handle specific functionalities.

#### **1. Embeddings: `embeddings.py`**

- Initializes the embeddings model using Google Generative AI Embeddings.

#### **2. Milvus Operations: `milvus.py`**

- Initializes the Milvus vector store.
- Provides functions to interact with Milvus.

#### **3. LangChain Operations: `langchain.py`**

- Sets up the QA chain using LangChain and the Gemini LLM.
- Defines the `format_docs` function to prepare documents for the QA chain.