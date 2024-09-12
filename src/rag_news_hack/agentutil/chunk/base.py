from abc import ABC

class BaseChunk(ABC):    
    @staticmethod
    def split(text: str, chunk_chars_length: int = 1500) -> list[str]:
        pass