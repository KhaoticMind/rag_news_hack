from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ConfigObject:
    type: str #The type of this object
    name: str #The name of this object
    instance: str #The class instance that this config object is associated with.
    metadata: dict #Additional information associated with this object
    created: Optional[int] = None #the epoch when this config object was created on the data store

class BaseStore(ABC):
    @abstractmethod
    def initialize(self, overwrite: bool =False):
        pass

    @abstractmethod
    def store_config(self, obj: ConfigObject) -> ConfigObject:
        pass

    @abstractmethod
    def get_config(self, object_type: str, object_name: str) -> Optional[ConfigObject]:
        pass

    @abstractmethod
    def get_entities(self, object_type: str) ->  List[tuple[str, int]]:
        pass