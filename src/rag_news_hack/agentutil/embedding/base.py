class BaseEmbedding:
    """
    Base class for getting text embeddings.
    This class implements the callable interface.
    """
    
    def __call__(self, text: str) -> list:
        """
        Call method to be overridden by subclasses for getting embeddings.
        
        :param text: The input text to be embedded.
        :return: A list representing the embedding of the input text.
        """
        raise NotImplementedError("Subclasses should implement this!")
    
    #TODO: CHANGE THIS!!!
    def get_embedding_dimension(self) -> int:
        return 1536