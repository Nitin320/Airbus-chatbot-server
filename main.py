from flask import Flask, request, jsonify
import os
import requests
import fitz  # PyMuPDF for PDF processing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set your Together AI API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "4112fdd91cc387561671f0d859fa17a239d249d8387ce05a009d5e48035bacfb")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
index = None
stored_texts = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to create FAISS index
def create_faiss_index(text_chunks):
    global index, stored_texts
    stored_texts = text_chunks
    embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return "\n".join([stored_texts[i] for i in indices[0]])

# Function to get AI response with context
def get_ai_response(prompt, context=""):
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    
    full_prompt = f"""
    You are an AI assistant specializing in Airbus A320 training materials. Provide concise and informative responses based on the given context.
    If the answer is not explicitly mentioned in the context, rely on your general knowledge to provide a reasonable response.
    Do not mention that the answer is missing or say "the context does not provide this information."
    
    Context: {context}
    
    User: {prompt}
    Assistant:
    """
    
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.7,
        "max_tokens": 300
    }
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json().get("choices")[0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

# Load and process PDF at startup
PDF_PATH = "a320training.pdf"  # Set your PDF file path here
if os.path.exists(PDF_PATH):
    extracted_text = extract_text_from_pdf(PDF_PATH)
    text_chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]
    create_faiss_index(text_chunks)

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"message": "The health check is successful"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "")
    context = retrieve_relevant_chunks(question) if index else ""
    response = get_ai_response(question, context)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
