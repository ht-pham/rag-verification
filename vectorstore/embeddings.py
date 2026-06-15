from numpy.linalg import norm


class NormalizedEmbeddings:

    def __init__(self, model_name="all-MiniLM-L6-v2", **kwargs):
        self.model_name = model_name
        self.kwargs = dict(kwargs)
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            backend_kwargs = dict(self.kwargs)
            model_kwargs = dict(backend_kwargs.pop("model_kwargs", {}))
            model_kwargs.setdefault("local_files_only", True)

            self._backend = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                **backend_kwargs,
            )

        return self._backend

    def embed_documents(self, documents):
        # Get the original embeddings
        original_embeddings = self._get_backend().embed_documents(documents)
        
        # Normalize each embedding vector to unit length
        normalized_embeddings = [self.normalize_vector(vec) for vec in original_embeddings]
        
        return normalized_embeddings
    
    def embed_query(self, query):
        # Get the original embedding for the query
        original_embedding = self._get_backend().embed_query(query)
        
        # Normalize the query embedding to unit length
        normalized_embedding = self.normalize_vector(original_embedding)
        
        return normalized_embedding

    def __call__(self, text):
        return self.embed_query(text)

    def normalize_vector(self, vec):
        vec_norm = norm(vec)
        if vec_norm == 0:
            return vec  # Return the original vector if its norm is zero to avoid division by zero
        return vec / vec_norm



