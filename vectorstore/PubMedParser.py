import json
import xml.etree.ElementTree as ET
from vectorstore.embeddings import NormalizedEmbeddings

class PubMedParser:
    def __init__(self, src: str,mesh_index='mesh_index.json',mesh_stats='mesh_counts.json',docs=[]):

        self.src = src
        self.mesh_index = mesh_index
        self.mesh_stats=mesh_stats
        self.docs = docs
    

    ### Task 1: Parse PubMed XML and extract relevant information
    def parse_xml(self,src=None):
        '''
        Definiton: Parses a PubMed XML file and extracts relevant information from each article.
        
        Returns:
            articles (list): A list of dictionaries, each containing information about an article, including:
                - title (str): The title of the article.
                - abstract (str): The abstract text of the article.
                - publication_type (str): The type of publication (e.g., "Journal Article").
                - publish_date (str): The publication date of the article.
                - mesh_terms (list): A list of MeSH terms associated with the article.
                - name_of_substance (str): A comma-separated string of substances mentioned in the article
        '''

        tree = ET.parse(src)
        root = tree.getroot()

        articles = []

        for article in root.findall(".//PubmedArticle"):
            # Extract the title
            title = article.findtext(".//ArticleTitle")
            # Extract the abstract
            abstract_texts = article.findall(".//AbstractText")
            abstract = " ".join([a.text for a in abstract_texts if a.text])
            
            # Extract the publication type
            publication_types = article.findall(".//PublicationType")
            journal_type = [pt.text for pt in publication_types if pt.text and "Journal Article" in pt.text]
            publication_type = journal_type[0] if journal_type else "Other"
            # Extract the publication date if any
            publish_date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "Unknown"

            # Extract the substances mentioned in the article
            substances = article.findall(".//NameOfSubstance")
            substances_list = [s.text for s in substances if s.text]
            substances = ", ".join(substances_list) if substances_list else "None"

            # Extract the MeSH terms associated with the article
            mesh_terms = [
                mesh.findtext(".//DescriptorName")
                for mesh in article.findall(".//MeshHeading")
                if mesh.findtext(".//DescriptorName")
            ]
            # do not add titles with no mesh_terms to the articles list, as they will not be useful for retrieval
            if mesh_terms == []:
                continue
            # Remove universal MeSH terms that are not specific to the article's content
            universal_terms = ['Animals','Humans','Male','Female','Adult','Middle Aged','Aged','Child','Infant','Adolescent','Pregnancy','Species Specificity',
                       'In Vitro Techniques','Methods','Time Factors','Chemical Phenomena','Kinetics','Hydrogen-Ion Concentration','Temperature']
            mesh_terms = [term for term in mesh_terms if term not in universal_terms]

            # Append the extracted information to the articles list
            articles.append({
                "title": title,
                "abstract": abstract,
                "publication_type": publication_type,
                "publish_date": publish_date,
                "mesh_terms": mesh_terms,
                "name_of_substance": substances
            })

        print(f"Parsed {len(articles)} articles from the XML file.")

        return articles

    def convertParsedArticlesToDocuments(self,articles):
        '''
        Definition: Converts the parsed articles into a format suitable for building a vector store, where each article is represented as a document containing its abstract and metadata.
        Args:
            - articles (list): A list of dictionaries, each containing information about an article, including MeSH terms.
        Returns:
            documents (list): A list of document objects, where each document contains the abstract and metadata of an article, and is suitable for building a vector store.
        '''
        from langchain_core.documents import Document
        
        universal_terms = ['Rabbits','Dogs','Mice','Rats','Animals','Humans','Male','Female','Adult',
                           'Age Factors','Middle Aged','Aged','Child','Infant','Adolescent','Pregnancy',
                           'Species Specificity','In Vitro Techniques','Methods','Time Factors',
                           'Chemical Phenomena','Australia','Follow-Up Studies','Kinetics','Hydrogen-Ion Concentration',
                           'Temperature','Oxygen Consumption', 'Pain','Clinical Trials as Topic', 
                           'Exercise Test', 'Heart Rate', 'Long-Term Care', 'Placebos']
        mesh_terms_filtered = ['Membrane Potentials', 'Microsomes, Liver', 'Enzyme Activation', 'Drug Stability', 
                'Protein Conformation', 'Adrenergic beta-Agonists', 'Graft vs Host Reaction', 'Dopamine', 
                'Biological Transport, Active', 'Muscle Contraction', 'Tyrosine 3-Monooxygenase', 
                'Adenosine Triphosphate', 'NAD', 'Heart Rate', 'Serotonin', 'Blood Pressure',
                'Anti-Anxiety Agents', 'Myocardium', 'Isoproterenol', 'Hypertension', 'Escherichia coli', 
                'Cyclic AMP', 'Propranolol', 'Drug Interactions', 'Antipsychotic Agents', 
                'Adrenergic beta-Antagonists', 'Structure-Activity Relationship','Transplantation, Homologous', 
                'Neurotransmitter Agents', 'Norepinephrine', 'NADP', 'Streptococcus pneumoniae']
        for art in articles:
            # if the article contains any of the universal MeSH terms, skip it 
            # as it may not provide specific information relevant to the query
            if any(term in universal_terms for term in art["mesh_terms"]):
                continue
            # if the article contains any of the selected MeSH terms, include it in the output abstract
            elif any(term in mesh_terms_filtered for term in art["mesh_terms"]) and art['abstract'] != '':
                #mesh_terms = ", ".join(art["mesh_terms"]) if art["mesh_terms"] else "None"
                content = f"{art['abstract']}"
            
                metadata = {
                    "title": art["title"],
                    "mesh_terms": art["mesh_terms"],
                    # "publication_types": art["publication_type"],
                    # "publish_date": art["publish_date"],
                    # "name_of_substance": art["name_of_substance"]
                }
                self.docs.append(Document(page_content=content, metadata=metadata))
        
        # Export Documents to JSON
        docs = self.exportDocsToJSON(self.docs,'data/documents.json')
        # docs = []
        # for d in self.docs:
        #     d.id = d.metadata["title"]
        #     doc = d.dict()
        #     docs.append(doc)
        
        # with open('data/documents.json','w') as file:
        #     json.dump(docs,file,indent=4) 

        # Export MeSH terms with its documents to JSON
        self.mesh_index = self.buildMeSHIndex(docs=docs)
        return self.docs
    
    def exportDocsToJSON(self,listOfDocs,file_path):
        docs = []
        for d in listOfDocs:
            d.id = d.metadata["title"]
            doc = d.dict()
            docs.append(doc)
        with open(file_path,'w') as file:
            json.dump(docs,file,indent=4) 
        
        return docs


    def chunkDocuments(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n\n","\n",". ","!"]
        )
        

        chunks = splitter.split_documents(self.docs)
        
        for i,chunk in enumerate(chunks):
            chunk.id = str(i+1)

        chunks_df = [chunk.model_dump() for chunk in chunks]
        with open('data/documents_chunk.json','w') as file:
            json.dump(chunks_df,file,indent=4) 

        if chunks_df != []:
            for i,chunk in enumerate(chunks):
                print(chunk.page_content)
                if i == 1:
                    break
            print("Finished chunking documents")

        return chunks
    
    def retrieveSimilarChunks(self,query,vector_store_path,k=5,search_again=False):
        #norm_query = embeddings.embed_query(query=query)
        retriever = self.loadVectorStore(vector_store_path)
        
        # Find related MeSH terms from the query
        related_terms = self.findRelatedMeSHTerms(query)
        # Retrieve more 20 related docs to filter from vectorstore and avoid out of range error when k is too large
        large_k = min(k + 20, len(retriever.index_to_docstore_id))
        
        # Retrieve relevant chunks based on the query only when there are related MeSH terms found in the query
        # results = [ (Document1,score1), (Document2,score2),...]
        if related_terms:
            # Start retrieving (k + 20) chunks
            results = retriever.similarity_search_with_score(
                query,
                k=large_k
            )
            # Filter results to only include those with matching MeSH terms
            filtered_results = [
                r for r in results 
                if any(term in r[0].metadata.get('mesh_terms', []) for term in related_terms)
            ]
            # Take the top k from the filtered results
            results = filtered_results
        else: # if no related MeSH terms, return context as "None" 
            return "None"
        
        titles = [r[0].metadata['title'] for r in results]
        if search_again==False:
            print(f"Retrieved {len(titles)} articles:")
            for i,title in enumerate(titles,start=1):
                print(f"{i}. {title}")
        else:
            print(f"Retrieved {len(titles)} articles after searching again.")
            for i, title in enumerate(titles[k:],start=k+1):
                print(f"{i}. {title}")
        

        # Compute cosine similarity scores for retrieved chunks
        # 0 = completely dissimilar, 1 = identical
        norm_results = []
        cos_sim_scores = []
        for doc, score in results:
            cos_sim_scores.append(1 - score / 2)
            norm_results.append((doc, score))

        context = ''
        relevant_chunks = []
        
        for i,r in enumerate(norm_results):
             
            # About L2 distance score: The smaller the score, the more similar the chunk is to the query.
            # 0.0 - 0.5: very similar
            # 0.5 - 1.5: strong match
            # 1.5 - 3.0: weak/moderate match
            # 3.0+: irrelevant
            # Cosine similarity score: 0: completely dissimilar, 0.5: neutral, 1: identical

            # # Print only half of the retrieved chunks if first time searching:
            if i < int(large_k/2) and search_again==False:
                print(f"L2 distance score: {r[1]:.4f};\t Cosine similarity score: {cos_sim_scores[i]:.4f}")
                print(f"Document: {r[0]}\n")
            elif i > k and search_again==True:
                print(f"L2 distance score: {r[1]:.4f};\t Cosine similarity score: {cos_sim_scores[i]:.4f}")
                print(f"Document: {r[0]}\n")

            # if cosine similarity score >= 0.6, consider it a relevant chunk and add into the context
            if cos_sim_scores[i] >= 0.6:
                parsed_chunk = r[0].page_content.replace("\n"," ").strip()
                relevant_chunks.append(parsed_chunk)


        context = '\n'.join(relevant_chunks)
        if context == "":
            return "None"
        
        return context
        

    def buildVectorStore(self,chunks,vector_store_path):

        from langchain_community.vectorstores import FAISS
        embeddings = NormalizedEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        vectorstore = FAISS.from_documents(
            chunks,
            embeddings,
        )
        doc_count = len(vectorstore.index_to_docstore_id)
        print(f"Total vectors in FAISS store: {doc_count}")
        

        # Save the FAISS index to local storage
        vectorstore.save_local(vector_store_path)
        #vectorstore.save_local("./data/pubmed_faiss_index")
        # update your data source to point to the new vector store path
        self.src = vector_store_path
        print(f"Vector store saved to: {vector_store_path}")

    
    def loadVectorStore(self,vector_store_path):
        #from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        # Load the FAISS index
        vectorstore = FAISS.load_local(
            vector_store_path, 
            NormalizedEmbeddings(model_name="all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
            )
        
        #return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return vectorstore
    
    def buildMeSHIndex(self,docs):
        '''
        Definition: Builds an index of MeSH terms to look up associated article titles.
        Args:
            - articles (list): A list of dictionaries, each containing information about an article, including MeSH terms.
        Returns:
            mesh_index (dict): A dictionary where keys are MeSH terms and values are lists of article titles associated with each term.
        '''
        self.mesh_index = {}
        for doc in docs:
            for term in doc["metadata"]["mesh_terms"]:
                if term not in self.mesh_index:
                    self.mesh_index[term] = []
                content = {"id":doc["id"],"abstract":doc["page_content"]}
                self.mesh_index[term].append(content)

        with open("analysis/mesh_index.json", "w") as f:
            json.dump(self.mesh_index, f,indent=4)
        print("MeSH index saved to mesh_index.json")

        return self.mesh_index

    def getNumberOfArticles(self):
        '''
        Definition: Prints the number of articles associated with each MeSH term in a specified range.
        Args:
            - mesh_index (dict): A dictionary where keys are MeSH terms and values are lists of article titles associated with each term.
            - start (int): The starting index for the range of MeSH terms to analyze.
            - end (int): The ending index for the range of MeSH terms to analyze.
        Returns:
            mesh_terms_list_sorted (dict): A sorted dictionary of MeSH terms and their associated article counts for the specified range.
        '''
        # Get the count of articles for each MeSH term and sort them by count
        mesh_terms_list = {term: len(titles) for term, titles in self.mesh_index.items()}
        # Sort the MeSH terms by article count in descending order
        mesh_terms_list_sorted = dict(sorted(mesh_terms_list.items(), key=lambda item: item[1], reverse=True))

        # Print the MeSH terms and their associated article counts for the specified range
        # print(f"{end-start} MeSH terms by article count:")
        # for term, count in list(mesh_terms_list_sorted.items())[start:end]:
        #     print(f"{term}: {count} articles")

        with open("analysis/mesh_counts.json", "w") as f:
            json.dump(mesh_terms_list_sorted, f,indent=4)
        print("MeSH counts saved to mesh_counts.json")

        return mesh_terms_list_sorted

    def findRelatedMeSHTerms(self,query):
        '''
        Definition: Finds MeSH terms that are related to a given query by checking for the presence of MeSH terms in the query.
        Args:
            - query (str): The input query for which to find related MeSH terms.
            - mesh_terms (list): A list of MeSH terms to check against the query.
        Returns:
            related_terms (list): A list of MeSH terms that are found in the query.
        '''
        self.mesh_index = json.load(open("analysis/mesh_index.json","r"))

        related_terms = []
        for term in self.mesh_index.keys():
            if term.lower() in query.lower():
                related_terms.append(term)
        return related_terms
    
    
    

