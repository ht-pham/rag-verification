# Retrieval-Augmented Generation (RAG) System with FAISS for healthcare research LLM 

## Abstract
This project implements a **Retrieval-Augmented Generation** pipeline using FAISS for vector search and transformer-based models for semantic understanding. The system retrieves relevant context from a document vectorstore and generates grounded responses to user queries. This project is designed for **research** and **reproducibility** evaluation only.

## Methodology
### Build FAISS vectorstore for Retrieval system
1. :books: Extract 30,000 articles' title, abstract, and metadata from PubMed library [1]. The file version used for this project is pubmed26n0001.xml. 
2. :microscope: Filter out articles with too broad or generic meSH terms and all articles without abstract, and save title, abstract, meSH terms of each article as a Document object. The final number of stored articles is 3,698.
3. :hocho: Split the abstracts into smaller chunks for each article with LangChain Text Splitter. The total number of chunks is 10,340 chunks or vectors
4. :file_folder: Convert these text chunks into dense numerical vectors with HuggingFace Embeddings all-MiniLM-L6-v2 model and store these dense vectors as a vector store with FAISS IndexFlatL2 for efficient similarity search with Euclidean distance (L2) scores between query and dense vectors.
5. :mag_right: Retrieve top-k relevant chunks and their similarity scores with similarity_search_with_score(q,k=n) [2]

### Transformer-based agents for two-staged handling user query
When a user asks domain-specific questions, then the system will begin retrieving from the vector store and return relevant chunks as context to be passed along with the user's question to transformers. There are two stages for handling user query, each of which is called an agent.

- Agent 1: Verifier or Verification to check the correlation between retrieved chunks and query using zero shot classification which labels the correlation score as 'contradiction','neutral', or 'entailment' (or support).

- Agent 2: Summarizer or Summarization (task="summarization",model_id="facebook/bart-large-mnli") to summarize the context when retrieved context supports or contradicts the query.

Both agents use Facebook's BART large mnli model. This is a denoising autoencoder. [3]

## Folder Directory
```
.
├── data/
│    └──pubmed_faiss_index/
│       ├── index.faiss
│       └── index.pkl
├── pipeline/
│    ├── agent.py
│    ├── summarization.py
│    └── verification.py
├── vectorstore/
│    └── PubMedParser.py
└── main.py
```

### Setup Instructions
#### 1. Clone the repository
```bash
git clone  https://github.com/ht-pham/rag-verification.git
cd rag-verification
```
#### 2. Create a virtual environment
**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux**
```bash
python -m venv venv
venv/bin/activate
```
#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. (Optional) Build the FAISS vector store
To build your own local vector store, you can download the .xml file from PubMed site [1], update your source URL to 'data/<your-file>.xml' in buildLocalDB() function in main.py, and uncomment the line calling that function under if __name__ == '__main__' statement.
The vectorstore is already available when you clone this GitHub repo.
#### 5. Run the program
From the directory, run

```bash
python main.py
```

## References
[1] [Index of /pubmed/updatefiles](https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/)

[2] [LangChain Similarity Search with Score](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/similarity_search_with_score)

[3] Yin, W., Hay, J. and Roth, D., 2019, November. Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP) (pp. 3914-3923). Access link: [link](https://huggingface.co/facebook/bart-large-mnli)