'''
this is a system check file
'''
import json

with open('data/documents.json','r') as file:
    docs = json.load(file)

print(len(docs))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the FAISS index
vectorstore = FAISS.load_local(
    "data/pubmed_faiss_index", 
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
    )

print(vectorstore.index.ntotal)
print(type(vectorstore.index))

docs = vectorstore.docstore._dict

for k,v in list(docs.items())[:5]:
    print(f"Key: {k}, Abstract: {v}")


index_to_doc = vectorstore.index_to_docstore_id
for i in range(5):
    id = index_to_doc[i]
    doc = docs[id]
    print(f"Document ID: {id}\nChunk: {doc.page_content}\n) ")
    print(f"\n\Title: {doc.metadata['title']}\n) ")
    print(f"\n\MeSH terms: {doc.metadata['mesh_terms']}\n) ")

print("="*100)
for i in range(5):
    id = index_to_doc[i]
    doc = docs[id]
    print(f"Document ID: {id}\n\nDocument: {doc}\n")
