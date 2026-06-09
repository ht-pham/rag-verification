import json

from vectorstore.PubMedParser import PubMedParser
from pipeline.extraction import Extractor
from pipeline.verification import Verifier
from pipeline.summarization import Summarizer

import random

#------------------- BUILDING LOCAL DATABASE ----------------------------
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
#------------------- END OF BUILDING LOCAL DATABASE ----------------------

#------------------- LOADING COMPONENTS' FUNCTIONS ------------------------------
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

def get_vectorstore():
    retriever = PubMedParser("data/pubmed26n0001.xml",'analysis/mesh_index.json','analysis/mesh_counts.json')
    vector_store_path = "data/pubmed_faiss_index"
    vectorstore = retriever.load_vectorstore(vector_store_path)
    return vectorstore, retriever

def load_components():
    # Load vectorstore
    vectorstore, rag = get_vectorstore()

    # Load all agents
    pipeline1 = Verifier("facebook/bart-large-mnli")
    #pipeline2 = Summarizer("google/flan-t5-large")
    pipeline2 = Summarizer("Qwen/Qwen2.5-3B-Instruct")
    agents = [pipeline1,pipeline2]

    return rag, agents

#------------------ END OF LOADING COMPONENTS' FUNCTION ------------------------

#------------------- PIPELINE FUNCTIONS ------------------------------
def run_generation_pipeline(query,context,agents):
    ''' When context is retrieved, run the rest of the pipelines '''

    answer = ""
    context = context.replace("\n"," ").strip()
    print(f"*** Extracted context: \n{context} ***\n\n")
    verified,score = agents[0].classify(query,context)
    print(f"*** Simple answer: {verified}, confident score: {score} ***")
    if verified == "Not enough information":
        answer = verified

    else:
        summary = agents[1].summarize(context.replace("\n"," ").strip())
        
        answer = verified + " because "+ summary
        #answer = verified + " because " + summary.replace("\n"," ").strip()
    
    return verified, answer

def recursive_retrieval(search_limit,query,k,rag,agents):

    times = range(1,search_limit+1)
    searches = {i:f"Search attempt #{i} " for i in times}

    search_count = 0
    while search_count < search_limit:
        context = rag.retrieve_similar_chunks(query,"data/pubmed_faiss_index",k,False)
        search_count += 1
        verified, answer = run_generation_pipeline(query,context,agents)

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
            print(f">>> {answer}") # "yes/no/maybe" + reason
            print("==============================")
            break
    
    return context, verified, answer
#------------------- END OF PIPELINE FUNCTIONS ------------------------------

#------------------- DEMOS FUNCTIONS ------------------------------
def test(search_limit):
    rag, agents = load_components()

    test_queries = [
        "Do beta-antagonists reduce heart rate?",
        "Do beta-blockers lower blood pressure?",
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
    context, verified, answer=recursive_retrieval(search_limit=search_limit,query=query,k=5,rag=rag,agents=agents)        
    # for q in test_queries:
    #     print(f"Question: {q}")
    #     recursive_retrieval(search_limit=3,query=q,k=5,rag=rag,agents=agents)

def run_demo(file_path='test/yesQA.json',output_path='test/yesQA_results.json',search_limit=10,k=5):

    rag, agents = load_components()

    test_queries = []
    contexts = []
    
    with open(file_path,'r') as file:
        all_qa = json.load(file)

    for mesh_term, qa in all_qa.items():
        print(f"MeSH Term: {mesh_term}")
        for q in qa["queries"]:
            print(f"Question: {q}")
            test_queries.append(q)
            context, verified, answer=recursive_retrieval(search_limit=search_limit,query=q,k=k,rag=rag,agents=agents)
            contexts.append({
                "mesh_term": mesh_term,
                "question": q,
                "verified_answer": verified,
                "final_answer": answer,
                "context": context}
            )

    with open(output_path,'w') as file:
        json.dump(contexts, file, indent=4)

def run_pubmedQA(file_path='test/pubmedQA_labeled.json',output_path='test/pubmedQA_results.json',search_limit=10,k=5):

    rag, agents = load_components()

    test_queries = []
    contexts = []
    
    with open(file_path,'r') as file:
        all_qa = json.load(file)

    for qa in all_qa:
        print(f"MeSH Term: {qa['context']['meshes']}")
        print(f"Question: {qa['question']}")
        test_queries.append(qa['question'])
        context, verified, answer=recursive_retrieval(search_limit=search_limit,query=qa['question'],k=k,rag=rag,agents=agents)
        contexts.append({
            "mesh_term": qa['context']['meshes'],
            "question": qa['question'],
            "verified_answer": verified,
            "final_answer": answer,
            "context": context}
        )

    with open(output_path,'w') as file:
        json.dump(contexts, file, indent=4)

if __name__ == "__main__":

    # Run this when need to rebuild the vectorstore
    #vectorstore, meSH_terms, meSH_terms_counts = build_local_db()
    #meSH_terms, meSH_terms_counts, qa, docs = load_all_data()
    # run a short test demo
    test(search_limit=5)

    # run demo with random selected question
    #run_demo(file_path='test/yesQA.json',output_path='test/yesQA_results.json',search_limit=3,k=5)
    #run_demo(file_path='test/noQA.json',output_path='test/noQA_results.json',search_limit=3,k=5)
    #run_pubmedQA(file_path='test/pqa_labeled.json',output_path='test/results/pqa_labeled_3R.json',search_limit=3,k=5)
    
    
    
            

   

    

    