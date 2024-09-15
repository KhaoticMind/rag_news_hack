from abc import abstractmethod
from collections import defaultdict
from inspect import Parameter, Signature
from typing import Annotated, Any, Dict, TypedDict

from ..ragstore import RAGData, RAGDatabase


class FunctionDict(TypedDict):
    name: str
    description: str
    code: str
    parameter_types: Dict[str, str]
    return_type: str


class BasicTool():
    def __init__(self, name: str, description: str):
        self.name = name
        self.__name__ = name
        self.description = description
        self.set_signature()

    @classmethod
    @abstractmethod
    def set_signature(cls):
        pass

    def get_description(self) -> str:
        return self.description

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def __call__(self, ** kwargs) -> str:
        pass


class DynamicFunctionBase:
    def __init__(self, function_dict: FunctionDict):
        self.name: str = function_dict["name"]
        self.description: str = function_dict["description"]
        self.code: str = function_dict["code"]
        self.parameter_types: dict = function_dict["parameter_types"]
        self.return_type: str = function_dict["return_type"]
        self.local_namespace: dict = {}

        # Compile the function code into the local namespace
        exec(self.code, globals(), self.local_namespace)

        # Retrieve the function object from the local namespace
        self.tool = self.local_namespace.get(self.name)

        if self.tool is None:
            raise ValueError(f"Function '{self.name}' could not be created.")


class RagTool(BasicTool):
    def __init__(self, rag_store: RAGDatabase, description="A tool that allows you to search a RAG database to snipets of html pages (from brazil and united states) based on the given query. It uses vector embedding to find the best matches for the passed queries. The returned result are the 10 most relavant snipets across all passed queries.", name='RagTool'):
        super().__init__(name=name, description=description)
        self.rag_store: RAGDatabase = rag_store

    @classmethod
    def set_signature(cls):
        param1 = Parameter('cls', kind=Parameter.POSITIONAL_OR_KEYWORD)
        param2 = Parameter('queries', kind=Parameter.KEYWORD_ONLY,
                           annotation=Annotated[list[str], 'A list of queries that will be searched for in the database. The returned results are the results best ranked (RRF) across all queries.'])
        params = [param1, param2]
        cls.__call__.__signature__ = Signature(
            parameters=params, return_annotation=Annotated[str, 'Informacoes recuperadas da base de RAG'])

    def __call__(self, **kwargs) -> str:
        res: list[str] = []
        data: dict[str, list[RAGData]] = {}
        k = 60

        queries: list[str] = kwargs.get('queries', [])
        for query in queries:
            data[query] = self.rag_store.query_text(query)

        #Lets get the result length, so we can return the same number of itens
        len_result = len(data[query])

        # Do RRF on the results
        # To store the RRF scores for each "ID"
        scores: dict[str, float] = defaultdict(float)
        # To store which lists each ID appeared in
        rank_map: dict[str, RAGData] = defaultdict()

        # Loop over each list of RAGData for each source
        for _, rag_list in data.items():
            for rank, rag in enumerate(rag_list):
                id = rag.metadata['id']                
                # Update the RRF score for the ID
                scores[id] += 1 / (k + rank + 1)
                rank_map[id] = rag

        # Sort IDs by their RRF scores (highest score first)
        sorted_ids = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True)

        # Create a final ranked list of RAGData objects
        final_data = [rank_map[id] for id in sorted_ids]

        for d in final_data[:len_result]:
            f = f"#URL:{d.metadata['url']}\n {d.data}\n\n"
            res.append(f)        
        return '\n\n'.join(res)
