from abc import ABC, abstractmethod

from pydantic import BaseModel


class LoadedData(BaseModel):
    content: str
    metadata: dict


class DocLoader(ABC):
    @abstractmethod
    def load(self, source, **kwargs) -> list[LoadedData]:
        pass
