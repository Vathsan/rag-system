"""
Vector Search Demo
Demonstrate semantic search using ChromaDB
"""

import chromadb
from sentence_transformers import SentenceTransformer

print("🔍 Vector Search Demo")
print("=" * 40)

# Initialize ChromaDB and model
client = chromadb.Client()
collection = client.create_collection("alliance_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Add sample documents
sample_docs = [
    "In machine learning, it is common to have to manage very large collections of files, meaning hundreds of thousands or more",
    "This page covers options to scale out traditional machine learning methods to very large datasets",
    "You cannot access a cloud without first having a cloud project",
    "Below there are instructions on starting a Windows VM or a Linux VM"
]

collection.add(
    documents=sample_docs,
    ids=[f"sample_{i+1}" for i in range(len(sample_docs))]
)

# Test vector search
query = "What is cognitive computing?"
results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: '{query}'")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - distance
    print(f"  {i+1}. Similarity: {similarity:.3f} - {doc}")

print("\n✅ Vector search demo completed!")
