from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson.objectid import ObjectId
from typing import List
from dataclasses import dataclass
from ..embedding import BaseEmbedding
from .base import RAGData, RAGDatabase

from copy import deepcopy
from uuid import uuid4

from .. import SecretRetriever


class AzureCosmosMongoRAGDatabase(RAGDatabase):
    """
    A class that extends RAGDatabase and integrates Azure CosmosDB MongoDB for document storage and retrieval.
    """

    def __init__(self, service_name: str, user: str, database_name: str, collection_name: str, embedding_function: BaseEmbedding, number_items_to_return: int = 5, max_distance: float = 0.8):
        """
        Initialize the AzureCosmosMongoRAGDatabase instance.

        Args:
            connection_string (str): CosmosDB MongoDB connection string.
            database_name (str): The name of the CosmosDB database.
            collection_name (str): The name of the CosmosDB collection.
            embedding_function (BaseEmbedding): Embedding function for generating text embeddings.
            number_items_to_return (int): The number of items to return in a query. Defaults to 5.
            max_distance (float): The maximum distance to return similar items. Defaults to 0.8.
        """
        super().__init__(embedding_function, number_items_to_return, max_distance)
        mongo_connection = 'mongodb+srv://{user}:{password}@{service_name}.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000'
        mongo_connection = mongo_connection.format(user=user, service_name=service_name, password=SecretRetriever.get_secret('AZ_COSMOS_MONGO_PWD'))

        self.client = MongoClient(mongo_connection)
        self.database_name = database_name
        self.db = self.client[database_name]
        self.collection_name = collection_name
        self.collection = self.db[collection_name]
        self._create_indexes()

    def _create_indexes(self):
        """
        Create indexes for efficient querying of embeddings and metadata.
        """
        self.collection.create_index(
            [("metadata.id", DESCENDING)], unique=True)

        self.db.command({
            'createIndexes': self.collection_name,
            'indexes': [
                {
                    'name': 'embedding_VectorSearchIndex',
                    'key': {
                        "embedding": "cosmosSearch"
                    },
                    'cosmosSearchOptions': {
                        'kind': 'vector-ivf',
                        'numLists': 1,
                        'similarity': 'COS',
                        'dimensions': 1536
                    }
                }
            ]
        })

    def save_text(self, text: str, metadata: dict):
        """
        Save the text and its corresponding embedding to CosmosDB MongoDB.

        Args:
            text (str): The text to be stored.
            metadata (dict): Additional metadata to be stored with the text.
        """
        embedding = self._calculate_embedding(text)
        meta = deepcopy(metadata)
        meta['id'] = str(metadata.get('id', uuid4()))
        document = {
            "data": text,
            "embedding": embedding,
            "metadata": meta
        }
        try:
            self.collection.insert_one(document)
        except DuplicateKeyError:
            # If a document with the same id exists, update it
            self.collection.update_one(
                {"metadata.id": meta.get('id')}, {"$set": document})

    def query_text(self, query_text: str) -> List[RAGData]:
        """
        Search for the most similar texts in the database based on the query text using vector similarity.

        Args:
            query_text (str): The text to search for.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        query_embedding = self._calculate_embedding(query_text)

        # Perform the vector similarity search (cosine similarity or other metrics)
        pipeline = [
            {
                "$search": {
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": self.number_items_to_return
                    }
                }
            }
        ]

        results = self.collection.aggregate(pipeline)
        rag_data_list = []

        for result in results:
            rag_data = RAGData(
                data=result["data"],
                distance=0,  # MongoDB does not calculate distance by default, you might need to implement this
                metadata=result["metadata"]
            )
            rag_data_list.append(rag_data)

        return rag_data_list

    def get(self, attributes: dict = {}) -> List[RAGData]:
        """
        Retrieve data from CosmosDB MongoDB based on a set of attributes.

        Args:
            attributes (dict): A dictionary of attributes to filter by.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text and metadata.
        """
        query = {"metadata." + key: value for key, value in attributes.items()}
        results = self.collection.find(query).limit(
            self.number_items_to_return)

        rag_data_list = []
        for result in results:
            rag_data = RAGData(
                data=result["data"],
                distance=0,  # No distance for a simple get operation
                metadata=result["metadata"]
            )
            rag_data_list.append(rag_data)

        return rag_data_list

    def close(self):
        """
        Close the database connection.
        """
        self.client.close()
