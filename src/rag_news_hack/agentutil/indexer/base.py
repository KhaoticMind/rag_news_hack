from typing import Callable, Optional

from ..chunk import BaseChunk
from ..loader import DocLoader, LoadedData
from ..ragstore import RAGDatabase


class BaseIndexer:
    def __init__(self,
                 loader: DocLoader,
                 rag_store: RAGDatabase,
                 chuncker: BaseChunk,
                 pre_chunker_handler: Optional[Callable[[
                     list[LoadedData]], list[LoadedData]]] = None,
                 pos_chunker_handler: Optional[Callable[[list[LoadedData]], list[LoadedData]]] = None):
        self.rag_store = rag_store
        self.loader = loader
        self.chunker = chuncker
        self.pre_chunker_handler = pre_chunker_handler
        self.pos_chunker_handler = pos_chunker_handler

    def index(self, source: str):
        result = self.loader.load(source=source)        

        if self.pre_chunker_handler is not None:
            result = self.pre_chunker_handler(result)

        for r in result:
            content = r.content
            metadata = r.metadata
            chunks: list[LoadedData] = []

            if self.chunker:
                splited_content = self.chunker.split(content)                

                for s in splited_content:
                    data = LoadedData(content=s, metadata=metadata)
                    chunks.append(data)
            else:
                chunks = [r]

            if self.pos_chunker_handler is not None:
                chunks = self.pos_chunker_handler(chunks)

            for c in chunks:                
                self.rag_store.save_text(c.content, c.metadata)
