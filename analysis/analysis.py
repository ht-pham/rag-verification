path = "./data/pubmed_faiss_index"

from langchain_community.vectorstores import FAISS 
from vectorstore.embeddings import NormalizedEmbeddings

vectorstore = FAISS.load_local(path, NormalizedEmbeddings(model_name="all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
        
index_type = vectorstore.index.__class__.__name__
vectorstore_name = vectorstore.docstore.__class__.__name__
print(f"Index type: {index_type}")
print(f"Vector store type: {vectorstore_name}")

num_vectors = vectorstore.index.ntotal
dimension = vectorstore.index.d
# Extract all vectors from the flat index
q = "heart disease"
indices = vectorstore.search(q,k=3,search_type="similarity")
for index in indices:
    print(index)
    print("\n")

import faiss
import numpy as np

index = faiss.read_index(path+"/index.faiss")

print("Number of vectors:", index.ntotal)
print("Vector dimension:", index.d)
print("Index type:", type(index))

vectors = np.vstack([
    index.reconstruct(i)
    for i in range(index.ntotal)
])

import pandas as pd

print(vectors.shape)
for vector in vectors[:5]:
    print(vector)
print(f"Min: {vectors[:5].min()}")
print(f"Max: {vectors[:5].max()}")
print(f"Mean: {vectors[:5].mean()}")

df = pd.DataFrame(vectors)

print(df.head())

# df.to_csv("./data/faiss_vectors.csv", index=False)