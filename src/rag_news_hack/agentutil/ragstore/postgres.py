import psycopg2
from psycopg2.extras import Json
from typing import List
from uuid import uuid4
from pgvector.utils import Vector

from .base import RAGData, RAGDatabase
from ..embedding import BaseEmbedding


class PostgresPgVectorRAGDatabase(RAGDatabase):
    """
    A class that extends RAGDatabase and integrates PostgreSQL with pgvector for document storage and retrieval.
    """

    def __init__(self, db_name: str, user: str, password: str, host: str, port: int, embedding_function: BaseEmbedding, number_items_to_return: int = 5, max_distance: float = 0.8):
        """
        Initialize the PostgresPgVectorRAGDatabase instance.

        Args:
            db_name (str): PostgreSQL database name.
            user (str): PostgreSQL username.
            password (str): PostgreSQL password.
            host (str): Database host address.
            port (int): Database port.
            embedding_function (BaseEmbedding): Embedding function for generating text embeddings.
            number_items_to_return (int): The number of items to return in a query. Defaults to 5.
            max_distance (float): The maximum distance to return similar items. Defaults to 0.8.
        """
        super().__init__(embedding_function, number_items_to_return, max_distance)
        self.connection = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        """
        Create the table for storing text, embeddings, and metadata if it doesn't exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS rag_data (
            id UUID PRIMARY KEY,
            data TEXT,
            embedding VECTOR(%s),  -- Adjust to match your embedding dimensions
            metadata JSONB
        );
        """
        embedding_dimension = self.embedding_function.get_embedding_dimension()
        self.cursor.execute(create_table_query, (embedding_dimension,))
        self.connection.commit()

    def reset_store(self):
        """
        Drop and recreate the table to reset the store.
        """
        drop_table_query = "DROP TABLE IF EXISTS rag_data;"
        self.cursor.execute(drop_table_query)
        self._create_table()

    def save_text(self, text: str, metadata: dict):
        """
        Save the text and its corresponding embedding to PostgreSQL with pgvector.

        Args:
            text (str): The text to be stored.
            metadata (dict): Additional metadata to be stored with the text.
        """
        # Generate embedding
        embedding = self._calculate_embedding(text)

        # Convert the list of embedding values into a string for the query
        embedding_str = str(embedding)

        # Generate UUID if not present
        id_value = str(metadata.get('id', uuid4()))

        # Insert into the database
        insert_query = """
        INSERT INTO rag_data (id, data, embedding, metadata)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE 
        SET data = EXCLUDED.data, embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata;
        """
        self.cursor.execute(
            insert_query, (id_value, text, embedding_str, Json(metadata)))
        self.connection.commit()

    def query_text(self, query_text: str) -> List[RAGData]:
        """
        Search for the most similar texts in the database based on the query text using vector similarity.

        Args:
            query_text (str): The text to search for.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        # Generate query embedding
        query_embedding = self._calculate_embedding(query_text)

        # Convert the list of embedding values into a string for the query
        query_embedding_str = str(query_embedding)

        # Perform the vector similarity search using pgvector and casting the array to the vector type
        vector_query = f"""
        SELECT id, data, metadata, rank() over (ORDER BY embedding <=> %(emb)s ) as rank
        FROM rag_data
        where (embedding <=> %(emb)s) <= %(max_distance)s
        ORDER BY  embedding <=> %(emb)s
        LIMIT %(double_nitens)s
        """

        fulltext_query = f"""
           SELECT id, data, metadata, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', data), query) DESC) as rank
                FROM rag_data, plainto_tsquery('english', %(query_text)s ) query
                WHERE query @@ to_tsvector('english', data)
                ORDER BY ts_rank_cd(to_tsvector('english', data), query) DESC
                LIMIT %(double_nitens)s
            """

        hybrid_query = f"""
        WITH vector_search AS (
            {vector_query}
        ),
        fulltext_search AS (
            {fulltext_query}
        )
        SELECT
            coalesce(vector_search.id, fulltext_search.id) as id,
            coalesce(vector_search.data, fulltext_search.data) as data,
            coalesce(vector_search.metadata, fulltext_search.metadata) as metadata
        FROM vector_search
        FULL OUTER JOIN fulltext_search ON vector_search.id = fulltext_search.id
        ORDER BY COALESCE(1.0 / (%(k)s + vector_search.rank), 0.0) + COALESCE(1.0 / (%(k)s + fulltext_search.rank), 0.0) DESC
        LIMIT %(nitens)s
        """

        self.cursor.execute(hybrid_query, {'max_distance': self.max_distance,
                            'emb': query_embedding_str, 'query_text': query_text, 'double_nitens': self.number_items_to_return * 2, 'nitens': self.number_items_to_return, 'k': 60})
        results = self.cursor.fetchall()

        # Process and return results
        rag_data_list = []
        for result in results:
            id, data, metadata = result
            metadata['id'] = id
            # print(data, distance)
            # print('---')
            rag_data = RAGData(
                data=data,
                distance=0,
                metadata=metadata
            )
            if rag_data.distance <= self.max_distance:
                rag_data_list.append(rag_data)

        return rag_data_list

    def get(self, attributes: dict = {}) -> List[RAGData]:
        """
        Retrieve data from PostgreSQL based on a set of attributes.

        Args:
            attributes (dict): A dictionary of attributes to filter by.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        filters = []
        values = []

        # Build SQL query filters based on the attributes
        for key, value in attributes.items():
            filters.append(f"metadata->>'{key}' = %s")
            values.append(value)

        # Create the query string
        filter_query = " AND ".join(filters) if filters else "TRUE"
        query = f"SELECT data, metadata FROM rag_data WHERE {filter_query} LIMIT %s;"
        self.cursor.execute(query, (*values, self.number_items_to_return))
        results = self.cursor.fetchall()

        # Process and return results
        rag_data_list = []
        for result in results:
            data, metadata = result
            rag_data = RAGData(
                data=data,
                distance=0,  # No distance for get operation
                metadata=metadata
            )
            rag_data_list.append(rag_data)
        return rag_data_list

    def close(self):
        """
        Close the database connection.
        """
        self.cursor.close()
        self.connection.close()
