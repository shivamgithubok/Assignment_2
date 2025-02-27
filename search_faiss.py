import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index
index = faiss.read_index("faiss_index.bin")

# Load the original documents
documents = np.load("documents.npy", allow_pickle=True)

# Load the same model used for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define a test query
query = ["What is CDP?"]

# Convert query to embedding
query_embedding = model.encode(query, convert_to_numpy=True)

# Search the FAISS index (k = number of results)
k = 3
distances, indices = index.search(query_embedding, k)

# Display results
print("\n🔎 Search Results:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]})")
