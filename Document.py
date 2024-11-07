import pdfplumber
import os
import numpy as np
import faiss
from flask import Flask, request, jsonify, send_from_directory, abort
import pickle
import logging
import requests
from azure.storage.blob import BlobServiceClient
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)

CORS(app)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

AZURE_CONNECTION_STRING = ""
CONTAINER_NAME = "pdf-files"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Azure OpenAI settings
AZURE_OPENAI_ENDPOINT = "https://ip-openai-serv.openai.azure.com"
AZURE_OPENAI_API_KEY = ""
CHAT_DEPLOYMENT_NAME = "gpt-4o-chat"
EMBEDDINGS_DEPLOYMENT_NAME = "text-embeddings-legal"

# Caching mechanisms
file_embeddings_cache = {}  # Cache for file embeddings
query_cache = {}  # Cache for query embeddings

# FAISS index and file management
index = None
paragraphs = None

@app.get("/list_pdfs/")
def list_pdfs():
    blobs = container_client.list_blobs()
    pdfs = [blob.name for blob in blobs]
    return pdfs

'''
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file.filename)
    blob_client.upload_blob(file.file.read(), overwrite=True)
    return {"filename": file.filename}

'''
@app.route('/upload_pdf/', methods=['POST'])
def upload():
    """Handle PDF file uploads."""
    if request.method == 'POST':
        f = request.files['file']
        print("File:"+f.filename)
        if not f.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        file_path = f.filename
        f.save(file_path)
        print("File saved locally.Going to save in blob")
        try:
            with open(file_path, "rb") as file_data:
                blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=file_path)
                print(f"Uploading to Azure Blob with blob name: {file_path}")
                blob_client.upload_blob(file_data, overwrite=True)
        except Exception as e:
            return jsonify({"error": f"Failed to upload file to Azure Blob Storage: {str(e)}"}), 500

        embeddings_path = f"{file_path}.pkl"

        # Load or create embeddings for the uploaded file
        global paragraphs, index
        print("going to create embedding")
        paragraphs, embedding_matrix = load_or_create_embeddings(file_path, embeddings_path)

        print("Embedding created. Creating index")
        # Create FAISS index for the uploaded file
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)

        # Cache the embeddings for the file to avoid reprocessing
        file_embeddings_cache[file_path] = {'paragraphs': paragraphs, 'index': index}
        print("Index saved in local cache. Returning file: "+f.filename)

        return {"filename": f.filename}



@app.route('/delete_pdf/<fileName>', methods=['DELETE'])
def delete_pdf(fileName):
    """Delete a PDF file from local storage and Azure Blob Storage if present."""

    # Ensure the filename ends with .pdf and sanitize the input
    if not fileName.endswith(".pdf"):
        return jsonify({"error": "Only PDF files can be deleted"}), 400

    # Create the local file path
    local_file_path = fileName
    pickle_file_name = fileName+".pkl"

    # Step 1: Delete the local file (if exists)
    if os.path.exists(local_file_path):
        try:
            os.remove(local_file_path)
            print(f"Deleted local file: {local_file_path}")
        except Exception as e:
            return jsonify({"error": f"Failed to delete local file: {str(e)}"}), 500
    else:
        print(f"Local file {local_file_path} does not exist.")

    if os.path.exists(pickle_file_name):
        try:
            os.remove(pickle_file_name)
            print(f"Deleted local file: {pickle_file_name}")
        except Exception as e:
            return jsonify({"error": f"Failed to delete local file: {str(e)}"}), 500
    else:
        print(f"Local file {local_file_path} does not exist.")
    # Step 2: Delete the file from Azure Blob Storage (if exists)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=fileName)

    try:
        blob_client.delete_blob()
        print(f"Deleted file {fileName} from Azure Blob Storage.")
    except ResourceNotFoundError:
        print(f"Blob {fileName} not found in Azure Blob Storage.")
    except Exception as e:
        return jsonify({"error": f"Failed to delete file from Azure Blob Storage: {str(e)}"}), 500

    return jsonify(
        {"message": f"File {fileName} deleted successfully from both local storage and Azure Blob Storage"}), 200


def extract_text_from_pdf(pdf_path, header_region=(0, 50, 595, 20)):
    """Extract text from a PDF document and return each page's text along with metadata."""
    data = []
    document_name = os.path.basename(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:  # Only add non-empty pages
                page_data = {
                    "text": text,
                    "metadata": {
                        "document_name": document_name,
                        "page_number": page_num,
                        "total_pages": total_pages
                    }
                }
                data.append(page_data)
    return data


def summarize_text(context, question):
    """Summarize the relevant sections of the document using GPT-4."""
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        'Content-Type': 'application/json'
    }

    prompt = (
        f"Given the following context, answer the question. "
        f"If you don't know the answer, say 'I am not able to resolve your query. Please rephrase your query or try to find the answer by going through the PDF manually.'.\n"
        f"{context}\n"
        f"Question: {question}\n"
        f"Answer (include page number and name of the document given in context if relevant):"
    )

    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview",
        headers=headers,
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": 2000}
    )

    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def create_embeddings(text):
    """Create embeddings for a given text."""
    headers = {
        "api-key": AZURE_OPENAI_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post(
        f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDINGS_DEPLOYMENT_NAME}/embeddings?api-version=2023-05-15",
        headers=headers,
        json={"input": text}
    )

    response.raise_for_status()
    return np.array(response.json()['data'][0]['embedding'])


def load_or_create_embeddings(pdf_path, embeddings_path):
    """Load or create embeddings for a given PDF, storing both text and metadata."""
    if os.path.exists(embeddings_path):
        # Load cached embeddings
        with open(embeddings_path, 'rb') as f:
            print("Embedding already existed. Loading it.")
            embeddings_data = pickle.load(f)
            paragraphs = embeddings_data['paragraphs']
            embedding_matrix = embeddings_data['embedding_matrix']
    else:
        # Extract text from PDF and generate embeddings
        print("Embedding does not exist. Creating it.")
        pdf_data = extract_text_from_pdf(pdf_path)
        paragraphs = pdf_data
        embeddings = [create_embeddings(p['text']) for p in paragraphs]
        print("Embeddings created.")
        embedding_matrix = np.vstack(embeddings).astype('float32')

        # Save embeddings to disk
        with open(embeddings_path, 'wb') as f:
            pickle.dump({'paragraphs': paragraphs, 'embedding_matrix': embedding_matrix}, f)
    print("Embedding written to file")

    return paragraphs, embedding_matrix

def split_text_into_chunks(text, chunk_size=500):
    """Split text into chunks of specified size."""
    # Create chunks of size `chunk_size`
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

@app.route("/hello")
def hello():
    print("Coming here")
    return "Hello from the Doc reader!"


@app.route('/chat', methods=['POST'])
def chat():
    """Handle question answering."""
    data = request.get_json()
    query = data.get('question')
    filename = data.get('filename')

    print(query)
    print(filename)

    embeddings_path = f"{filename}.pkl"

    # Load or create embeddings for the uploaded file
    global paragraphs, index
    print("going to create embedding")
    paragraphs, embedding_matrix = load_or_create_embeddings(filename, embeddings_path)

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)

    # Cache the embeddings for the file to avoid reprocessing
    file_embeddings_cache[filename] = {'paragraphs': paragraphs, 'index': index}

    # Ensure the file has been uploaded and cached
    if filename not in file_embeddings_cache:
        print("file not found")
        return jsonify({"error": "File not found. Please upload the file first."}), 400

    print("Filename:"+filename)
    # Retrieve the cached paragraphs and FAISS index for the file
    paragraphs = file_embeddings_cache[filename]['paragraphs']
    index = file_embeddings_cache[filename]['index']

    print("paragraphs and index found")

    # Create query embedding (do not cache)
    query_embedding = create_embeddings(query).astype('float32').reshape(1, -1)

    # Search for the top 3 most relevant sections
    k = 3
    distances, indices = index.search(query_embedding, k)

    # Extract relevant sections along with their metadata
    relevant_sections = []
    for i in range(k):
        section_data = paragraphs[indices[0][i]]  # This now includes both text and metadata
        distance = distances[0][i]
        relevant_sections.append({
            "section": section_data['text'],
            "metadata": section_data['metadata'],  # Include metadata in the response
            "distance": float(distance)
        })

    # Summarize the relevant sections
    context = ' '.join([rs['section'] for rs in relevant_sections])
    answer = summarize_text(context, query)
    if 'I am not able to resolve your query' in answer :
        relevant_sections= []
    # Return the response, including metadata for each relevant section
    response = {
        "question": query,
        "answer": answer,
        "relevant_sections": relevant_sections
    }

    return jsonify(response)


# Configure the directory where your files are stored
FILE_DIRECTORY = 'C:/Users/jainprak/Desktop/Backend'

@app.route('/files/<path:filename>', methods=['GET'])
def serve_file(filename):
    try:
        return send_from_directory(FILE_DIRECTORY, filename)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':

    app.run(debug=True)
