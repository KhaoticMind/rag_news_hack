import pyodbc
from typing import List, Dict
from uuid import uuid4
from .base import RAGData, RAGDatabase
from ..embedding import BaseEmbedding
from .. import SecretRetriever
from copy import deepcopy
import json


class AzureSQLRAGDatabase(RAGDatabase):
    """
    A class that extends RAGDatabase and integrates Azure SQL Server for document storage and retrieval.
    """

    def __init__(self, server: str, database: str, username: str, embedding_function: BaseEmbedding, number_items_to_return: int = 5, max_distance: float = 0.8):
        """
        Initialize the AzureSQLRAGDatabase instance.

        Args:
            server (str): Azure SQL Server name.
            database (str): Azure SQL Database name.
            username (str): Azure SQL Server username.            
            embedding_function (BaseEmbedding): Embedding function for generating text embeddings.
            number_items_to_return (int): The number of items to return in a query. Defaults to 5.
            max_distance (float): The maximum distance to return similar items. Defaults to 0.8.
        """
        super().__init__(embedding_function, number_items_to_return, max_distance)
        password = SecretRetriever.get_secret('AZ_SQL_SERVER_PWD')
        self.connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"        
        self.connection = pyodbc.connect(self.connection_string, autocommit=True)
        self.cursor = self.connection.cursor()
        self._create_table()

    def _create_table(self):
        """
        Create the table for storing text, embeddings, and metadata if it doesn't exist.
        """
        create_table_query = """
IF OBJECT_ID(N'dbo.rag_data', N'U') IS NULL
create table dbo.rag_data
(
    id UNIQUEIDENTIFIER constraint pk__data primary key,
    data nvarchar(4000),
    metadata nvarchar(4000),
    embedding nvarchar(max)
);

IF OBJECT_ID(N'rag_embeddings', N'U') IS NULL
create table dbo.rag_embeddings
(
    id UNIQUEIDENTIFIER references dbo.rag_data(id),
    vector_value_id int,
    vector_value float
);

IF NOT EXISTS (SELECT *
FROM sys.indexes
WHERE name = 'csi__rag_embeddings ' AND object_id = OBJECT_ID('dbo.rag_embeddings'))
create clustered columnstore index csi__rag_embeddings 
    on dbo.rag_embeddings order (id);

if not exists(select *
from sys.fulltext_catalogs
where [name] = 'FullTextCatalog')
begin
    create fulltext catalog [FullTextCatalog] as default;
end;

IF NOT EXISTS (SELECT 1
FROM sys.fulltext_indexes
WHERE object_id = OBJECT_ID('dbo.rag_data'))
create fulltext index on dbo.rag_data (data) key index pk__data;

alter fulltext index on dbo.rag_data enable; 
"""
        create_function_query = """
create or alter function dbo.similar_documents(@vector nvarchar(max), @nitens int)
returns table
as
return 
with
    cteVector
    as
    (
        select
            cast([key] as int) as [vector_value_id],
            cast([value] as float) as [vector_value]
        from
            openjson(@vector)
    ),
    cteSimilar
    as
    (
        select top (@nitens)
            v2.id,
            1-sum(v1.[vector_value] * v2.[vector_value]) / 
        (
            sqrt(sum(v1.[vector_value] * v1.[vector_value])) 
            * 
            sqrt(sum(v2.[vector_value] * v2.[vector_value]))
        ) as cosine_distance
        from
            cteVector v1
            inner join
            dbo.rag_embeddings v2 on v1.vector_value_id = v2.vector_value_id
        group by
        v2.id
        order by
        cosine_distance
    )
select
    rank() over (order by r.cosine_distance asc) as rank,
    r.id,
    r.cosine_distance
from
    cteSimilar r;
        """
        self.cursor.execute(create_table_query)
        self.cursor.execute(create_function_query)
        self.connection.commit()

    def save_text(self, text: str, metadata: dict):
        """
        Save the text and its corresponding embedding to Azure SQL.

        Args:
            text (str): The text to be stored.
            metadata (dict): Additional metadata to be stored with the text.
        """
        # Generate embedding
        embedding = self._calculate_embedding(text)
        embedding_str = json.dumps(embedding)
        meta = deepcopy(metadata)
        meta['id'] = str(metadata.get('id', uuid4()))


        # Insert into the database
        insert_query = """
                DECLARE @id UNIQUEIDENTIFIER = ?;
                DECLARE @text nvarchar(4000) = ?;
                DECLARE @embedding nvarchar(max) = ?;
                DECLARE @metadata nvarchar(4000) = ?;
                INSERT INTO dbo.rag_data (id, data, embedding, metadata) VALUES (@id, @text, @embedding, @metadata);
                INSERT INTO dbo.rag_embeddings SELECT @id, CAST([key] AS INT), CAST([value] AS FLOAT) FROM OPENJSON(@embedding);
        """
        self.cursor.execute(insert_query, meta['id'], text, embedding_str, json.dumps(metadata))
        self.connection.commit()

    def query_text(self, query_text: str) -> List[RAGData]:
        """
        Perform a hybrid search by combining vector similarity search and keyword search.

        Args:
            query_text (str): The text to search for.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        k = 60
        # Generate query embedding

        query_embedding = self._calculate_embedding(query_text)
        query_embeddings_str = json.dumps(query_embedding)


        # First, perform vector similarity search
        hybrid_search = f"""
         DECLARE @k INT = ?;
         DECLARE @text NVARCHAR(4000) = ?;
         DECLARE @embedding NVARCHAR(max) = ?;         
            WITH keyword_search AS (
                SELECT
                    id,
                    data,
                    metadata,
                    rank() over (order by ftt.[RANK] desc) AS rank
                FROM 
                    dbo.rag_data 
                INNER JOIN 
                    FREETEXTTABLE(dbo.rag_data, data, @text, {self.number_items_to_return * 2}) AS ftt ON dbo.rag_data.id = ftt.[KEY]
            ),
            semantic_search AS
            (
                SELECT 
                    d.id, 
                    d.data,
                    d.metadata,
                    s.rank                    
                FROM 
                    dbo.similar_documents(@embedding, {self.number_items_to_return * 2}) AS s
                INNER JOIN 
                    dbo.rag_data AS d on s.id = d.id                
            )
            SELECT TOP({self.number_items_to_return})
                COALESCE(ss.id, ks.id) AS id,
                COALESCE(ss.data, ks.data) AS data,
                COALESCE(ss.metadata, ks.metadata) AS metadata,
                COALESCE(1.0 / (@k + ss.rank), 0.0) +
                COALESCE(1.0 / (@k + ks.rank), 0.0) 
                AS score -- Reciprocal Rank Fusion (RRF) 
            FROM
                semantic_search ss
            FULL OUTER JOIN
                keyword_search ks ON ss.id = ks.id
            ORDER BY 
                score DESC
        """
        self.cursor.execute(hybrid_search, k, query_text, query_embeddings_str)
        results = self.cursor.fetchall()
        
        # Combine vector and text search results
        rag_data_list = []
        for result in results:
            id, data, metadata, _ = result
            rag_data = RAGData(data=data, distance=0, metadata={"id": id, **json.loads(metadata)})
            rag_data_list.append(rag_data)
    

        return rag_data_list

    def get(self, attributes: Dict = {}) -> List[RAGData]:
        """
        Retrieve data from Azure SQL based on a set of attributes.

        Args:
            attributes (dict): A dictionary of attributes to filter by.

        Returns:
            List[RAGData]: A list of RAGData objects containing the matched text, similarity score, and metadata.
        """
        filters = []
        values = []
        for key, value in attributes.items():
            filters.append(f"JSON_VALUE(metadata, '$.{key}') = ?")
            values.append(value)

        filter_query = " AND ".join(filters) if filters else "1=1"
        query = f"SELECT top {self.number_items_to_return} data, metadata FROM rag_data WHERE {filter_query} ;"        
        self.cursor.execute(query, values)
        results = self.cursor.fetchall()

        rag_data_list = []
        for result in results:
            data, metadata = result
            rag_data = RAGData(data=data, distance=0, metadata=json.loads(metadata))
            rag_data_list.append(rag_data)
        return rag_data_list

    def close(self):
        """
        Close the database connection.
        """
        self.cursor.close()
        self.connection.close()
