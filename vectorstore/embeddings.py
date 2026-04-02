from numpy.linalg import norm
from langchain_huggingface import HuggingFaceEmbeddings


class NormalizedEmbeddings(HuggingFaceEmbeddings):

    def embed_documents(self, documents):
        # Get the original embeddings
        original_embeddings = super().embed_documents(documents)
        
        # Normalize each embedding vector to unit length
        normalized_embeddings = [self.normalize_vector(vec) for vec in original_embeddings]
        
        return normalized_embeddings
    
    def embed_query(self, query):
        # Get the original embedding for the query
        original_embedding = super().embed_query(query)
        
        # Normalize the query embedding to unit length
        normalized_embedding = self.normalize_vector(original_embedding)
        
        return normalized_embedding

    def normalize_vector(self, vec):
        vec_norm = norm(vec)
        if vec_norm == 0:
            return vec  # Return the original vector if its norm is zero to avoid division by zero
        return vec / vec_norm



