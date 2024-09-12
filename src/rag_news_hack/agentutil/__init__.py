import importlib
import os
from copy import deepcopy
from typing import Callable

from .datastore import BaseStore, ConfigObject


def instantiate_from_config(config: ConfigObject, store: BaseStore):
    # Import the module using the type, use relative import to know where the module is located
    module = importlib.import_module("." + config.type, 'agentutil')

    # Get the class from the module using the instance name
    cls = getattr(module, config.instance)

    params = deepcopy(config.metadata)
    for k in params.keys():
        # lets see if the parameter is a reference to another object, the format is #|:obj_type:obj_name:|#'
        if isinstance(params[k], str) and params[k].startswith('#|:') and params[k].endswith(':|#'):
            ref = params[k].split(':')
            config = store.get_config(ref[1], ref[2])
            obj = instantiate_from_config(config, store)
            params[k] = obj

    # Instantiate the class with the metadata as parameters
    instance = cls(**params)

    return instance


class SecretRetriever():

    @staticmethod
    def env_secret_retriever(name: str) -> str:
        return os.environ[name]

    retriver: Callable[[str], str] = env_secret_retriever

    @classmethod
    def set_retriever(cls, retriever=Callable[[str], str]):
        cls.retriver = retriever

    @classmethod
    def get_secret(cls, name: str) -> str:
        return cls.retriver(name)
