import os
import fitz  # PyMuPDF for PDF processing
import faiss  # FAISS for vector search
import numpy as np
import pickle  # To save/load data
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths for storing FAISS index and text chunks
FAISS_INDEX_FILE = "faiss_index.bin"
TEXT_CHUNKS_FILE = "text_chunks.pkl"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to create and save FAISS index
def create_faiss_index(text_chunks):
    embeddings = embedding_model.encode(text_chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save FAISS index and text chunks
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(TEXT_CHUNKS_FILE, "wb") as f:
        pickle.dump(text_chunks, f)

    print("✅ FAISS index and text chunks saved successfully.")

# Train on the PDF
PDF_PATH = "a320training.pdf"  # Set your PDF file path here
if os.path.exists(PDF_PATH):
    print("📖 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(PDF_PATH)
    text_chunks = [extracted_text[i:i+500] for i in range(0, len(extracted_text), 500)]
    
    print("🧠 Training FAISS index...")
    create_faiss_index(text_chunks)
else:
    print("❌ Error: PDF file not found.")
