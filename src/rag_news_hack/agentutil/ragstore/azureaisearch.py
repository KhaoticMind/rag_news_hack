from copy import deepcopy
from json import loads
from typing import List
from uuid import uuid4

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration,
                                                   SearchableField,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SearchIndex, SimpleField,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from azure.search.documents.models import VectorizedQuery

from ..embedding import BaseEmbedding
from .base import RAGData, RAGDatabase
from .. import SecretRetriever


class AzureSearchRAGDatabase(RAGDatabase):
    """
    A class that extends RAGDatabase and integrates Azure AI Search for document storage and retrieval.
    """

    def __init__(self, service_name: str, index_name: str, embedding_function: BaseEmbedding, number_items_to_return: int = 5, max_distance: float = 0.8):
        """
        Initialize the AzureSearchRAGDatabase instance.

        Args:
            service_name (str): Azure Search service name.
            api_key (str): Azure Search API key.
            index_name (str): Azure Search index name.
            embedding_function (BaseEmbedding): Embedding function for generating text embeddings.
            number_items_to_return (int): The number of items to return in a query. Defaults to 5.
            max_distance (float): The maximum distance to return similar items. Defaults to 0.8.
        """
        super().__init__(embedding_function, number_items_to_return, max_distance)
        self.service_name = service_name
        self.api_key = SecretRetriever.get_secret('AZ_AI_SEARCH_KEY')
        self.index_name = index_name
        self.endpoint = f"https://{service_name}.search.windows.net"
        self.credentials = AzureKeyCredential(self.api_key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credentials)

        # Try to create the index if it not exists
        self._create_index()

    def _create_index(self):
        try:
            self.index_client.get_index(self.index_name)
        except ResourceNotFoundError as e:

            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String,
                            key=True, sortable=True, filterable=True, facetable=True),
                SearchableField(name="data", type=SearchFieldDataType.String),
                SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                            searchable=True, vector_search_dimensions=self.embedding_function.get_embedding_dimension(), vector_search_profile_name="myHnswProfile"),
            ]

            # Configure the vector search configuration
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="myHnsw"
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ]
            )

            # Initialize the index client
            # Create the search index with the semantic settings
            index = SearchIndex(name=self.index_name, fields=fields,
                                vector_search=vector_search)

            self.index_client.create_or_update_index(index)
        finally:
            # And now get the client
            self.client = SearchClient(
                endpoint=self.endpoint, index_name=self.index_name, credential=self.credentials)

    def reset_store(self):
        self.index_client.delete_index(self.index_name)
        self._create_index()

    def save_text(self, text: str, metadata: dict):
        """
        Save the text and its corresponding embedding to Azure Search.

        Args:
            text (str): The text to be stored.
            metadata (dict): Additional metadata to be stored with the text.
        """

        # Lets copy all the metadata, since these will become "fields"
        d: dict = deepcopy(metadata)
        # And save the text and the embedding
        d['data'] = text
        d['embedding'] = self._calculate_embedding(text)
        # If we have an ID make sure its a str
        if 'id' in metadata:
            d['id'] = str(metadata['id'])
        else:
            d['id'] = str(uuid4())

        # Index the documents
        try:
            result = self.client.merge_or_upload_documents(documents=[d])
        except HttpResponseError as e:
            # Lets see if the error was the a given column doesnt exist, if its the case lets create the field and retry the insertion
            if 'does not exist on type' in str(e):
                self._update_fields(d)
                result = self.client.merge_or_upload_documents(documents=[d])
            else:
                raise e

    def query_text(self, query_text: str) -> List[RAGData]:
        """
        Search for the most similar texts in the database based on the query text.

        Args:
            query_text (str): The text to search for.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        search_embed = self._calculate_embedding(query_text)
        vector_query = VectorizedQuery(
            vector=search_embed,
            # kind='vector',
            fields="embedding",
            exhaustive=True,
            k_nearest_neighbors=self.number_items_to_return * 2,
            weight=0.5
        )

        # Perform search
        results = self.client.search(
            top=self.number_items_to_return,
            search_text=query_text,
            vector_queries=[vector_query]
        )

        # Process results
        rag_data_list = []
        for result in results:
            metadata = {}
            for k, v in result.items():
                if not (str(k).startswith('@')) and k != 'data':
                    metadata[k] = v
            rag_data = RAGData(
                data=result['data'],
                # as per https://learn.microsoft.com/en-us/azure/search/vector-search-ranking
                # (1.0 - result['@search.score']) / result['@search.score'],
                distance=result['@search.score'],
                metadata=metadata
            )
            # Lets just consider elements that are not too far away
            if rag_data.distance <= self.max_distance:
                rag_data_list.append(rag_data)
        return rag_data_list

    def get(self, attributes: dict = {}) -> List[RAGData]:
        """
        Retrieve data from the Azure Search index based on a set of attributes.

        Args:
            attributes (dict): A dictionary of attributes to filter by.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """        

        # Build the search filter query using equality for each attribute
        filters = []
        for key, value in attributes.items():
            if isinstance(value, str):
                # Escape single quotes in string values to avoid query errors
                value = value.replace("'", "''")
                filters.append(f"{key} eq '{value}'")
            else:
                filters.append(f"{key} eq {value}")

        # Join all filters with 'and'
        filter_query = " and ".join(filters)

        try:
            # Search with the filter query
            results = self.client.search(
                search_text="*",  # search all documents
                filter=filter_query,
                top=self.number_items_to_return
            )

            # Process results
            rag_data_list = []
            for result in results:
                metadata = {}
                for k, v in result.items():
                    if not (str(k).startswith('@')) and k != 'data':
                        metadata[k] = v
                rag_data = RAGData(
                    data=result['data'],
                    distance=result['@search.score'],
                    metadata=metadata
                )
                rag_data_list.append(rag_data)

            return rag_data_list
        except HttpResponseError as e:
            print(f"Search query failed: {e}")
            return []


    def close(self):
        """
        Close the database connection.
        """
        # No specific action needed to close Azure Search connection
        pass

    def _update_fields(self, dictionary):
        # Get the current index schema
        current_index = self.index_client.get_index(self.index_name)
        current_field_names = [field.name for field in current_index.fields]

        # List to store new fields
        new_fields = []

        # Loop through the dictionary and check if fields already exist
        for key, value in dictionary.items():
            if key not in current_field_names:
                field_type = AzureSearchRAGDatabase._get_azure_search_data_type(
                    value)
                new_fields.append(SimpleField(
                    name=key, type=field_type, searchable=False, filterable=True, facetable=True))

        # If there are new fields, update the index
        if new_fields:
            print(f"Adding new fields: {[field.name for field in new_fields]}")
            # Append new fields to existing schema
            current_index.fields.extend(new_fields)
            self.index_client.create_or_update_index(current_index)
            print("Index updated successfully!")
        else:
            print("No new fields to add.")

    def _get_azure_search_data_type(value):
        if isinstance(value, str):
            return SearchFieldDataType.String
        elif isinstance(value, int):
            return SearchFieldDataType.Int32
        elif isinstance(value, float):
            return SearchFieldDataType.Double
        elif isinstance(value, bool):
            return SearchFieldDataType.Boolean
        elif isinstance(value, list):
            # Adjust based on list contents
            return SearchFieldDataType.Collection(SearchFieldDataType.String)
        else:
            return SearchFieldDataType.String  # Fallback
