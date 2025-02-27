import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the JSON data
with open("cleaned_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract text data from JSON
documents = data["CDP_Documentation"].split(". ")  # Split into sentences

# Load a pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")  # Small, fast model

# Convert text documents to embeddings
document_embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index (L2 normalization improves search)
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
index.add(document_embeddings)  # Add embeddings to FAISS index

# Save FAISS index
faiss.write_index(index, "faiss_index.bin")
np.save("documents.npy", np.array(documents))

print("✅ FAISS index built and saved!")
