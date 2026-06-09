path = "./data/pubmed_faiss_index"

from langchain_community.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings

vectorstore = FAISS.load_local(path, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)
        
index_type = vectorstore.index.__class__.__name__
vector_dimension = vectorstore.index_to_docstore_id
print(f"Index type: {index_type}")
print(f"Vector dimension: {vector_dimension}")