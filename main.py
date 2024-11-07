import io
import PyPDF2
from fastapi import FastAPI, File, UploadFile, HTTPException
from azure.storage.blob import BlobServiceClient
# from pymupdf import fitz
import requests
# import fitz  # PyMuPDF
from PyPDF2 import PdfFileReader
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_headers=["*"],
    allow_methods=["*"]
)


# Configure the directory where your files are stored
FILE_DIRECTORY = 'C:/Users/jainprak/Desktop/Backend'

@app.route('/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    try:
        return send_from_directory(FILE_DIRECTORY, filename)
    except FileNotFoundError:
        abort(404)

# Azure Blob Storage settings
AZURE_CONNECTION_STRING = ""
CONTAINER_NAME = "pdf-files"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = "https://ip-openai-serv.openai.azure.com"  # e.g., https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY = "b7acb25a4fa347f5ad6d3cdc3d14f447"
CHAT_DEPLOYMENT_NAME = "gpt-4-legal" # Deployment name for chat model
EMBEDDINGS_DEPLOYMENT_NAME = "text-embeddings-legal"  # Deployment name for embeddings model


# Headers for Azure OpenAI requests
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY
}

# Function to create embeddings using Azure OpenAI
async def create_embeddings(text):
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDINGS_DEPLOYMENT_NAME}/embeddings?api-version=2023-05-15",
        headers=headers,
        json={"input": text}
    )
    response.raise_for_status()
    return response.json()['data']

# Function to handle chat using Azure OpenAI
async def chat_with_model(prompt):
    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT_NAME}/completions?api-version=2023-03-15-preview",
        headers=headers,
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150
        }
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file.filename)
    blob_client.upload_blob(file.file.read(), overwrite=True)

    await file.seek(0)

    # Create and store embeddings
    embedding = await create_embedding(file)
    document_id = file.filename  # Unique identifier for the document
    store_embedding(document_id, embedding)  # Store embedding in Cosmos DB

    return {"filename": file.filename}


async def create_embedding(file: UploadFile):
    # Extract text from PDF
    text = ""
    with pdfplumber.open(await file.read()) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Create embeddings
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()

    return embeddings.flatten().tolist()
    
def store_embedding(document_id: str, embedding: list):
    # Prepare the item to be stored in Cosmos DB
    item = {
        "id": document_id,
        "embedding": embedding,
    }
    try:
        container.create_item(item)
    except exceptions.CosmosHttpResponseError as e:
        raise HTTPException(status_code=500, detail=f"Error storing embedding: {e.message}")


@app.get("/list_pdfs/")
async def list_pdfs():
    blobs = container_client.list_blobs()
    pdfs = [blob.name for blob in blobs]
    return {"pdfs": pdfs}

@app.post("/create_embeddings/")
async def create_embeddings_from_pdf(filename: str):
    try:
        # Download the PDF from Azure Blob Storage
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=filename)
        pdf_data = blob_client.download_blob().readall()

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_data)

        # Create embeddings for the extracted text
        embeddings = await create_embeddings(text)

        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
def extract_text_from_pdf(pdf_data: bytes) -> str:
    text = ""
    with PyPDF2.PdfFileReader(io.BytesIO(pdf_data)) as pdf:
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            text += page.extract_text() or ""
    return text

@app.post("/chat/")
async def chat(query: str):
    response = await chat_with_model(query)
    return {"response": response}

# Run the app using: uvicorn main:app --reload
