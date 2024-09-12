from abc import ABC, abstractmethod
from typing import List, Any
from ..embedding import BaseEmbedding
from dataclasses import dataclass

@dataclass
class RAGData:
    """
    A data class representing the result of a RAG (Retrieval-Augmented Generation) query.

    Attributes:
        data (str): The retrieved text content.
        distance (float): The similarity distance between the query and the retrieved content.
        metadata (dict): Additional metadata associated with the retrieved content.
    """
    data: str
    distance: float
    metadata: dict

class RAGDatabase(ABC):
    """
    An abstract base class for RAG (Retrieval-Augmented Generation) databases.

    This class provides a common interface for different RAG database implementations.
    """

    def __init__(self, embedding_function: BaseEmbedding, number_items_to_return: int = 5, max_distance: float = 0.8):
        """
        Initialize the RAGDatabase instance.

        Args:
            embedding_function: A function that takes a text string as input and returns its embedding.
            number_items_to_return (int): The number of items to return in a query. Defaults to 5.
        """
        self.embedding_function = embedding_function
        self.number_items_to_return = number_items_to_return
        self.max_distance = max_distance

    def _calculate_embedding(self, text: str) -> List[float]:
        """
        Calculate the embedding for the given text using the provided embedding function.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding of the text.
        """
        return self.embedding_function(text)

    @abstractmethod
    def save_text(self, text: str, metadata: dict):
        """
        Save the text and its corresponding embedding to the database.

        Args:
            text (str): The text to be stored.
            metadata (dict): Additional metadata to be stored with the text.
        """
        pass

    @abstractmethod
    def query_text(self, query_text: str) -> List[RAGData]:
        """
        Search for the most similar texts in the database based on the query text.

        Args:
            query_text (str): The text to search for.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        pass

    @abstractmethod
    def get(self, attributes : dict = {}) -> List[RAGData]:
        """
        Retrieve data from the database based on a set of attributes.
        Args:
            attributes (dict): A dictionary of attributes to filter by.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the database connection.
        """
        pass