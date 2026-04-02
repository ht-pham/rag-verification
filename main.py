import json

from vectorstore.PubMedParser import PubMedParser
from pipeline.extraction import Extractor
from pipeline.verification import Verifier
from pipeline.summarization import Summarizer



def old_generate_function(query,context):
    ''' PAST FUNCTION - NOT FOR USE ANYMORE
    KEPT FOR RECORD PURPOSES ONLY'''
    def verifyAnswer(query,context,answer):
        llm = load_llm()

        prompt = f"""
                Act as a verifier. Check on the answer from this LLM model to verify if the answer is fully supported by the evidence.
                
                Instructions:
                - Read the provided context as the sole evidence(s) for the LLM's answer.
                - Veify if the answer is fully supported by the evidence(s).
                - Return your answer by saying "SUPPORTED" or "NOT_SUPPORTED".
                
                CONTEXT
                {context}

                QUESTION:
                {query}

                ANSWER:
                {answer}

                Return:
                - SUPPORTED
                - NOT_SUPPORTED
                """

        answer = llm.generate([prompt])
        #print("TYPE:", type(answer))
        #print("DIR:", dir(answer)[:80])
        #print("REPR:", repr(answer)[:500])
        return answer.generations[0][0].text
    def load_llm():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from langchain_huggingface import HuggingFacePipeline
        #from langchain_community.llms import HuggingFacePipeline
        import torch
        from transformers import pipeline

        model_id = "google/flan-t5-large"
        #model_id = "google/medgemma-1.5-4b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    llm = load_llm()

    prompt = f"""
            You are a biomedical research assistant. 
            
            Instructions:
            - Use ONLY the provided CONTEXT.
            - The context contains multiple documents labeled [1], [2], [3] 
            - You must find explicit evidence before answering.
            - Do NOT guess.
            
            Rules:
            - If NO supporting evidence is found, then Answer = "Not enough information" AND Evidence="None"
            - If evidence supports the statement, then Answer = "Yes"
            - If evidence contradicts the statement, then Answer = "No"
            - You MUST provide both Answer and Evidence.
            - Evidence MUST be copied EXACTLY from the context.

        
            CONTEXT
            {context}

            QUESTION:
            {query}

            Step 1: Identify relevant sentences from the context.
            Step 2: If no relevant evidence, say "Not enough information"

            Output format:
            Answer: <Yes / No / Not enough information>
            Evidence: <copy the exact supporting sentence(s) from context OR "None">

            Final answer:
            """

    answer = llm.generate([prompt])
    #print("TYPE:", type(answer))
    #print("DIR:", dir(answer)[:80])
    #print("REPR:", repr(answer)[:500])
    return answer.generations[0][0].text

def loadAllData():
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

def buildLocalDB():
    ''' RUN THIS FUNCTION ONLY WHEN NEED TO REBUILD THE LOCAL VECTORSTORE DATABASE'''
    # Load and parse PubMed XML data
    parser = PubMedParser("data/pubmed26n0001.xml",'analysis/mesh_index.json','analysis/mesh_counts.json')
    articles = parser.parse_xml(parser.src)
    print(f"Total articles parsed: {len(articles)}")
    # Chunk documents and build vector store
    docs = parser.convertParsedArticlesToDocuments(articles)
    print(f"Total documents created: {len(docs)}")
    chunks = parser.chunkDocuments()
    print(f"Total chunks created: {len(chunks)}")
    # build vector store and save to disk
    vector_store_path = "data/pubmed_faiss_index"
    mesh_terms = parser.buildVectorStore(chunks, vector_store_path)
    mesh_terms_count = parser.getNumberOfArticles()
    # load vector store and create retriever
    vectorstore = parser.loadVectorStore(vector_store_path, k=5)
    return vectorstore, mesh_terms, mesh_terms_count

def import_pipelines():
    pipeline1 = Extractor("google/flan-t5-large")
    pipeline2 = Verifier("facebook/bart-large-mnli")
    pipeline3 = Summarizer("facebook/bart-large-mnli")

    return pipeline1, pipeline2, pipeline3

def getVectorstore():
    retriever = PubMedParser("data/pubmed26n0001.xml",'analysis/mesh_index.json','analysis/mesh_counts.json')
    vector_store_path = "data/pubmed_faiss_index"
    vectorstore = retriever.loadVectorStore(vector_store_path)
    return vectorstore, retriever

def runInitialCheck(query="",retriever=PubMedParser,meSH_terms=set()):
    related_terms = retriever.findRelatedMeSHTerms(query,meSH_terms)
    print(f"Question:\n{query}")
    if related_terms == []:
        print(">>> No MeSH terms found for this question. Unable to retrieve relevant chunks.")
        print("==============================")
        return "None"
    
    print("Related MeSH terms:", related_terms)
    return related_terms

def searchContext(q,vectorstore,related_terms,n=5):
    # Retrieve relevant chunks based on the query
    # results = [ (Document1,score1), (Document2,score2),...]
    results = vectorstore.similarity_search_with_score(
        q,
        k=n,
        #filter={"mesh_terms": {"$in": related_terms}}
    )  # Retrieve top-k chunks and their L2 distance scores for the question
    
    titles = [r[0].metadata['title'] for r in results]
    print(f"Retrieved articles: {titles}")

    context = ''
    relevant_chunks = []
    
    for i,r in enumerate(results,start=1):
        # this line is for showing the L2 distance score between the query and the retrieved chunk. 
        # The smaller the score, the more similar the chunk is to the query.
        # 0.0 - 0.5: very similar
        # 0.5 - 1.5: strong match
        # 1.5 - 3.0: weak/moderate match
        # 3.0+: irrelevant
        print(f"L2 distance score: {r[1]:.4f}\nDocument: {r[0]}\n")
        
        # Because chunks are sorted ASCENDING by distance, thresholding = 1.50
        # When chunk's L2 distance is too large, break the loop
        if r[1] < 1.51:
            parsed_chunk = r[0].page_content.replace("\n"," ").strip()
            relevant_chunks.append(parsed_chunk)
        else:
            # skip irrelevant chunks
            break

    context = '\n'.join(relevant_chunks)
    if context == "":
        return "None"
    
    return context

def run_pipelines(query,context,agents):
    answer = ""
    
    #output = agents[0].generate(query,context)
    if context == "None":
        answer = "Not enough information"
    else:
        print(f"*** Extracted context: \n{context} ***\n\n")
        verified,score = agents[1].classify(query,context)
        print(f"*** Simple answer: {verified}, confident score: {score} ***")
        if verified == "Not enough information":
            answer = verified
        else:
            raw_summary, refined_summary = agents[2].summarize(context.replace("\n"," ").strip())
            print(f"*** Initial summary: {raw_summary} ***")
            answer = verified + " because "+refined_summary
    return answer

def session(status,q,rag,vectorstore,agents):
    ''' WORK IN PROGRESS - NOT FOR USE YET'''
    def exitSession(q):
        if q.lower() == "exit":
            print("Exiting the program. Goodbye!")
            return False
    

    def searchAgain(count):
        if count >=3:
            print(">>> Multiple unsuccessful retrieval attempts. Please consider rephrasing your question or providing more specific details.")
            count = 0
            q = input("Enter your biomedical question (or type 'exit' to quit): ")
            if exitSession(q):
                return "inactive"
        
        return q, count
    
    if exitSession(q):
        return "inactive"
   
    search_again = 0
    while status=="active":
        # Check if there is any relevant meSH terms in the query
        found = runInitialCheck(q,rag)
        if found == "None": 
            search_again += 1 # if not, ask user to rephrase
            print(">>> The LLM model could not find sufficient evidence to answer the question based on the retrieved context.")
            print(">>> Please try rephrasing the question or providing more specific details.")
            q = input("Enter your biomedical question (or type 'exit' to quit): ")
            if exitSession(q):
                return "inactive"
            if search_again >=3:
                print(">>> Multiple unsuccessful retrieval attempts. Please consider rephrasing your question or providing more specific details.")
                search_again = 0
                q = input("Enter your biomedical question (or type 'exit' to quit): ")
                if exitSession(q):
                    return "inactive"
            continue
            
        context = searchContext(q,vectorstore,5)
        if context == "None":

            print(">>> No relevant documents found for the question.")
            print(">>> Please rephrase the question or try a new one.")
            continue
        # This line to check the final context to be passed to LLM
        
        answer = run_pipelines(q,context,agents)
    
    return status
            
    

if __name__ == "__main__":
    # Load all agents
    extractor,verifier,summarizer = import_pipelines()
    agents = [extractor,verifier,summarizer]

    # Run this when need to rebuild the vectorstore
    #vectorstore, meSH_terms, meSH_terms_counts = buildLocalDB()

    # Load vectorstore
    vectorstore, rag = getVectorstore()

    meSH_terms, meSH_terms_counts, qa, docs = loadAllData()


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
    
    
    # mesh_terms = {'Membrane Potentials', 'Microsomes, Liver', 'Enzyme Activation', 'Drug Stability', 
    #               'Protein Conformation', 'Adrenergic beta-Agonists', 'Graft vs Host Reaction', 'Dopamine', 
    #               'Biological Transport, Active', 'Muscle Contraction', 'Tyrosine 3-Monooxygenase', 
    #               'Adenosine Triphosphate', 'NAD', 'Heart Rate', 'Serotonin', 'Blood Pressure',
    #                 'Anti-Anxiety Agents', 'Myocardium', 'Isoproterenol', 'Hypertension', 'Escherichia coli', 
    #                 'Cyclic AMP', 'Propranolol', 'Drug Interactions', 'Antipsychotic Agents', 
    #                 'Adrenergic beta-Antagonists', 'Structure-Activity Relationship','Transplantation, Homologous', 
    #                 'Neurotransmitter Agents', 'Norepinephrine', 'NADP', 'Streptococcus pneumoniae'}
    
    for q in test_queries:
        # Check if there is any relevant meSH terms in the query
        related_terms = runInitialCheck(q,rag,meSH_terms)
        if related_terms == "None": # if not, move on 
            continue
        context = rag.retrieveSimilarChunks(q,"data/pubmed_faiss_index",5)
        #context = searchContext(q,vectorstore,related_terms,5)
        if context == "None":
            print(">>> No relevant documents found for the question.")
            print(">>> Please rephrase the question or try a new one.")
            continue
        # This line to check the final context to be passed to LLM
        
        answer = run_pipelines(q,context,agents)
        if answer == "Not enough information":
            print(">>> The LLM model could not find sufficient evidence to answer the question based on the retrieved context.")
            print(">>> Please try rephrasing the question or providing more specific details.")
        else:
            print(f">>> {answer}")
        print("==============================")

   

    

    