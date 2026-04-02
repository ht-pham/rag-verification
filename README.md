# Retrieval-Augmented Generation for healthcare research LLM 

### RAG
Retrieve from FAISS Vectorstore built from PubMed.xml file [1]
Retrieval function: similarity_search_with_score(q,k=n) to return top-k relevant documents and its L2 distance [2]


### LLM
- Agent 1: Verification (task="zero-shot-classification",model_id="facebook/bart-large-mnli") to check the relation between documents and query
- Agent 2: Summarization (task="summarization",model_id="facebook/bart-large-mnli") to summarize the context when documents supports or contradicts the query


### References
[1] [Index of /pubmed/updatefiles](https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/)
[2] [LangChain Similarity Search with Score](https://reference.langchain.com/python/langchain-qdrant/vectorstores/Qdrant/similarity_search_with_score)