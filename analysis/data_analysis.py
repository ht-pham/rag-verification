from vectorstore.old_pubmed import getNumberOfArticles, parse_pubmed_xml, buildMeshIndex, collectMeshTerms
#from vectorstore.pubmed import google_mesh_clusters, gpt_mesh_clusters

global google_mesh_clusters
google_mesh_clusters= {
    "Neurobiology_Signaling": [
        "Neurotransmitter Agents", "Dopamine", "Norepinephrine", 
        "Serotonin", "Tyrosine 3-Monooxygenase", "Cyclic AMP", 
        "Membrane Potentials", "Anti-Anxiety Agents", "Antipsychotic Agents"
    ],
    "Cardiovascular_Pharmacology": [
        "Hypertension", "Blood Pressure", "Heart Rate", "Myocardium", 
        "Adrenergic beta-Antagonists", "Propranolol", 
        "Adrenergic beta-Agonists", "Isoproterenol", "Muscle Contraction"
    ],
    "Bioenergetics_Molecular": [
        "Adenosine Triphosphate", "NADP", "NAD", "Oxidation-Reduction", 
        "Oxygen Consumption", "Protein Conformation", "Protein Binding", 
        "Binding Sites", "Enzyme Activation", "Structure-Activity Relationship", 
        "Biological Transport, Active", "Osmolar Concentration"
    ],
    "Clinical_Translation_Immunology": [
        "Graft vs Host Reaction", "Transplantation, Homologous", 
        "Clinical Trials as Topic", "Drug Evaluation", "Drug Interactions", 
        "Drug Stability", "Escherichia coli", "Streptococcus pneumoniae", 
        "Microsomes, Liver"
    ]
}
global gpt_mesh_clusters
gpt_mesh_clusters = {
    
    "Neuropharmacology_Monoamine_Regulation": [
        "Dopamine",
        "Serotonin",
        "Norepinephrine",
        "Tyrosine 3-Monooxygenase",
        "Neurotransmitter Agents",
        "Anti-Anxiety Agents",
        "Antipsychotic Agents",
        "Histamine H1 Antagonists"
    ],
    
    "Cardiovascular_Beta_Adrenergic_Axis": [
        "Hypertension",
        "Blood Pressure",
        "Heart Rate",
        "Myocardium",
        "Muscle Contraction",
        "Adrenergic beta-Antagonists",
        "Adrenergic beta-Agonists",
        "Propranolol",
        "Isoproterenol",
        "Norepinephrine"
    ],
    
    "Molecular_Energy_Signaling_Pathways": [
        "Adenosine Triphosphate",
        "NAD",
        "NADP",
        "Cyclic AMP",
        "Enzyme Activation",
        "Protein Conformation",
        "Membrane Potentials",
        "Biological Transport, Active"
    ],
    
    "Pharmacodynamics_Drug_Mechanism": [
        "Dose-Response Relationship, Drug",
        "Drug Interactions",
        "Drug Stability",
        "Structure-Activity Relationship"
    ],
    
    "Organ_Specific_Physiology_Metabolism": [
        "Liver",
        "Microsomes, Liver",
        "Kidney",
        "Brain",
        "Muscles",
        "Erythrocytes",
        "gamma-Glutamyltransferase"
    ],
    
    "Infectious_Disease_Immunology": [
        "Escherichia coli",
        "Streptococcus pneumoniae",
        "Graft vs Host Reaction",
        "Transplantation, Homologous"
    ]
}

def analyze_mesh_clusters(mesh_index, group_of_clusters,name):
    cluster_counts = {cluster: sum(mesh_index.get(term, []).__len__() for term in terms) for cluster, terms in group_of_clusters.items()}
    total_articles = sum(cluster_counts.values())
    print(f"\n{name} MeSH Clusters Article Counts:")
    for cluster, count in cluster_counts.items():
        print(f"{cluster}: {count} articles")
    print(f"Total articles in {name} clusters: {total_articles}")
    return cluster_counts, total_articles

def analyze_overlap(clusters1, clusters2):
    terms1 = set(term for terms in clusters1.values() for term in terms)
    terms2 = set(term for terms in clusters2.values() for term in terms)
    overlap_terms = terms1.intersection(terms2)
    print(f"\nNumber of overlap MeSH terms: {len(overlap_terms)}")
    print(f"Overlap MeSH terms: {overlap_terms}")
    return overlap_terms


# Build MeSH index for quick lookup
articles = parse_pubmed_xml("./data/pubmed26n0001.xml")
mesh_index = buildMeshIndex(articles)

#getNumberOfArticles(mesh_index, start=0, end=100)
articles_per_cluster_google, total_google = analyze_mesh_clusters(mesh_index, google_mesh_clusters,"Google")
articles_per_cluster_gpt, total_gpt = analyze_mesh_clusters(mesh_index, gpt_mesh_clusters,"GPT")
#global overlap_terms
overlap_terms = analyze_overlap(google_mesh_clusters, gpt_mesh_clusters)
