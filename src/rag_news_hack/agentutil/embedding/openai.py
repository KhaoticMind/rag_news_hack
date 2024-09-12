from openai import OpenAI
from .base import BaseEmbedding

class OpenAIEmbedding(BaseEmbedding):
    """
    Class to get the embedding of a given text using an OpenAI embedding model.
    """
    
    def __init__(self, model: str = 'text-embedding-ada-002'):
        """
        Initialize the TextEmbedding class with a specific model.
        
        :param model: The embedding model to use, e.g., 'text-embedding-ada-002'.
        """
        self.model = model
    
    def __call__(self, text: str) -> list[float]:
        """
        Get the embedding of a given text using an OpenAI embedding model.
        
        :param text: The input text to be embedded.
        :return: A list representing the embedding of the input text.
        """
        try:
            client = OpenAI()
            #input_text = text.replace("\n", " ")
            response = client.embeddings.create(input=text, model=self.model)
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

