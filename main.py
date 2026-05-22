import json

from vectorstore.PubMedParser import PubMedParser
from pipeline.extraction import Extractor
from pipeline.verification import Verifier
from pipeline.summarization import Summarizer

import random

def load_all_data():
    ''' Only meSH_terms is used. The remaining files are for future use '''
    with open('analysis/mesh_index.json','r') as file:
        meSH_terms = json.load(file)

    with open('analysis/mesh_counts.json','r') as file:
        meSH_terms_counts = json.load(file)

    with open('test/testQA.json','r') as file:
        qa = json.load(file)
    with open('data/documents.json','r') as file:
        docs = json.load(file)
    
    return meSH_terms, meSH_terms_counts, qa, docs

def build_local_db():
    ''' RUN THIS FUNCTION ONLY WHEN NEED TO REBUILD THE LOCAL VECTORSTORE DATABASE'''
    # Load and parse PubMed XML data
    parser = PubMedParser("data/pubmed26n0001.xml",'analysis/mesh_index.json','analysis/mesh_counts.json')
    articles = parser.parse_xml(parser.src)
    print(f"Total articles parsed: {len(articles)}")
    # Chunk documents and build vector store
    docs = parser.convert_articles_to_documents(articles)
    print(f"Total documents created: {len(docs)}")
    chunks = parser.chunk_documents()
    print(f"Total chunks created: {len(chunks)}")
    # build vector store and save to disk
    vector_store_path = "data/pubmed_faiss_index"
    mesh_terms = parser.build_vectorstore(chunks, vector_store_path)
    mesh_terms_count = parser.get_number_of_articles()
    # load vector store and create retriever
    vectorstore = parser.load_vectorstore(vector_store_path, k=5)
    return vectorstore, mesh_terms, mesh_terms_count

def import_pipelines():
    pipeline1 = Extractor("facebook/bart-large-cnn")
    pipeline2 = Verifier("facebook/bart-large-mnli")
    #pipeline3 = Summarizer("google/flan-t5-large")
    pipeline3 = Summarizer("Qwen/Qwen2.5-3B-Instruct")
    
    return pipeline1, pipeline2, pipeline3

def get_vectorstore():
    retriever = PubMedParser("data/pubmed26n0001.xml",'analysis/mesh_index.json','analysis/mesh_counts.json')
    vector_store_path = "data/pubmed_faiss_index"
    vectorstore = retriever.load_vectorstore(vector_store_path)
    return vectorstore, retriever

def runInitialCheck(query="",retriever=PubMedParser,meSH_terms=set()):
    related_terms = retriever.find_related_mesh_terms(query,meSH_terms)
    print(f"Question:\n{query}")
    if related_terms == []:
        print(">>> No MeSH terms found for this question. Unable to retrieve relevant chunks.")
        print("==============================")
        return "None"
    
    print("Related MeSH terms:", related_terms)
    return related_terms


def run_generation_pipeline(query,context,agents):
    ''' When context is retrieved, run the rest of the pipelines '''

    answer = ""
    print(f"*** Extracted context: \n{context} ***\n\n")
    verified,score = agents[1].classify(query,context)
    print(f"*** Simple answer: {verified}, confident score: {score} ***")
    if verified == "Not enough information":
        answer = verified

    else:
        summary = agents[2].summarize(context.replace("\n"," ").strip())
        print(f"*** Initial summary: {summary} ***")
        answer = verified + " because "+ summary
    
    return answer

def recursive_retrieval(search_limit,query,k,rag):
    times = range(1,search_limit+1)
    searches = {i:f"Search attempt #{i} " for i in times}

    search_count = 0
    while search_count < search_limit:
        context = rag.retrieve_similar_chunks(query,"data/pubmed_faiss_index",k,False)
        search_count += 1
        answer = run_generation_pipeline(query,context,agents)

        if (context == "None") or (answer == "Not enough information"):
            print(f"*** {searches[search_count]} failed to retrieve relevant documents for the question. ***")
            if search_count == search_limit:
                print(">>> The LLM model could not find sufficient evidence to answer the question based on the retrieved context.")
                print(">>> Please consider rephrasing the question or providing more specific details.")
                print("==============================")
            else:
                print("*** Expanding search...Trying to search again... ***")
                k += 20
            
        elif answer != "Not enough information":
            print(f">>> {answer}")
            print("==============================")
            break


            
    

if __name__ == "__main__":
    # Load all agents
    extractor,verifier,summarizer = import_pipelines()
    agents = [extractor,verifier,summarizer]

    # Run this when need to rebuild the vectorstore
    #vectorstore, meSH_terms, meSH_terms_counts = build_local_db()

    # Load vectorstore
    vectorstore, rag = get_vectorstore()

    meSH_terms, meSH_terms_counts, qa, docs = load_all_data()

    test_queries = [
        "Do beta-antagonists reduce heart rate?",
        "Do beta-blockers lower blood pressure?",
        "Can drugs degrade over time?",
        "Do anti-anxiety agents affect neurotransmitter activity?",
        "Is structure-activity relationship important in drug design?",
        "Is C-reactive protein a reliable biomarker for cardiovascular disease?",
        "Is isoproterenol used in cardiac treatment?",
        "Does immunotherapy improve survival outcomes in patients with melanoma?",
        "Does regular physical activity reduce the risk of developing type 2 diabetes?",
        "Is hypertension associated with high blood pressure?"
    ]
    
    random_query_index = random.randint(0, len(test_queries)-1)
    query = test_queries[random_query_index]
    print(f"Selected question: {query}")
    recursive_retrieval(search_limit=10,query=query,k=5,rag=rag)
    
    for q in test_queries:
        print(f"Question: {q}")
        recursive_retrieval(search_limit=3,query=q,k=5,rag=rag)
    
            

   

    

    