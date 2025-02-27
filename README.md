# FAISS-Based Searchable Index

## 📌 Project Overview
This project builds a **searchable knowledge base** using **FAISS (Facebook AI Similarity Search)** and **Sentence Transformers**. It takes **unstructured data**, converts it into **vector embeddings**, and enables **fast semantic search**.

## 🔥 Features
- **Text Embedding:** Uses `all-MiniLM-L6-v2` from `sentence-transformers`.
- **FAISS Indexing:** Efficient similarity search using `IndexFlatL2`.
- **JSON Data Processing:** Extracts and processes unstructured data.
- **Fast Querying:** Retrieves the most relevant information in milliseconds.

## 📂 Project Structure
```
📁 FAISS_Searchable_Index/
│── cleaned_data.json   # Raw data in JSON format
│── build_faiss.py      # Script to create the FAISS index
│── search_faiss.py     # Script to test and query FAISS
│── faiss_index.bin     # Saved FAISS index
|----pythonquestion-answer-system  
│── documents.npy       # Saved document texts
│── README.md           # Project documentation
```

## 🚀 Approach
1. **Load JSON Data:** Extracts text from `cleaned_data.json`.
2. **Generate Embeddings:** Converts text into numerical vectors using `SentenceTransformer`.
3. **Build FAISS Index:** Stores embeddings in FAISS for efficient similarity search.
4. **Query Processing:** Converts user queries to embeddings and retrieves the most relevant results.
5. **Display Results:** Outputs matching text snippets with similarity scores.

## 🔧 Setup & Installation
### 1️⃣ Install Dependencies
```sh
pip install faiss-cpu sentence-transformers numpy
```

### 2️⃣ Run FAISS Indexing Script
```sh
python build_faiss.py
```
✅ This generates `faiss_index.bin` and `documents.npy`.

### 3️⃣ Run Search Script
```sh
python search_faiss.py
```
✅ Enter queries and get relevant results!

## 🎯 Example Usage
### Build FAISS Index (`build_faiss.py`)
```python
import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open("cleaned_data.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    documents = data["CDP_Documentation"].split(". ")

# Encode using Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = model.encode(documents, convert_to_numpy=True)

# Create FAISS index
index = faiss.IndexFlatL2(document_embeddings.shape[1])
index.add(document_embeddings)
faiss.write_index(index, "faiss_index.bin")
numpy.save("documents.npy", np.array(documents))
print("✅ FAISS index built!")
```

### Query FAISS (`search_faiss.py`)
```python
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

index = faiss.read_index("faiss_index.bin")
documents = np.load("documents.npy", allow_pickle=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("🔍 Enter your search query: ")
query_embedding = model.encode([query], convert_to_numpy=True)

k = 3
distances, indices = index.search(query_embedding, k)

print("\n🔎 Top Results:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents[idx]} (Score: {distances[0][i]})")
```

## 📌 Next Steps
- **Fine-tune embeddings** using domain-specific models.
- **Improve FAISS search** by using `IVF`, `HNSW`, or `PQ` indexing.
- **Deploy as an API** using **FastAPI** or **Flask**.
- **Build a chatbot** that queries FAISS dynamically.

---
🚀 **Enjoy fast, efficient search with FAISS!** Let me know if you need improvements. 🔥

