import os
import json
import time
from typing import Optional, List
from .base import BaseStore, ConfigObject

class JSONStore(BaseStore):
    """
    A class to handle storage and retrieval of configuration objects in a directory as JSON files.
    Each file is named <type>_<name>.json, containing the instance, metadata, and created attributes.

    Attributes:
        directory (str): Path to the directory where JSON files are stored.
    """

    def __init__(self, directory: str):
        """
        Initializes the JSONStore with the specified directory path.

        Args:
            directory (str): The directory where the JSON files will be stored.
        """
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def initialize(self, overwrite: bool = False):
        """
        Initializes the store by optionally clearing all existing JSON files.
        
        Args:
            overwrite (bool): If True, clears all files in the directory.
        """
        if overwrite:
            for filename in os.listdir(self.directory):
                file_path = os.path.join(self.directory, filename)
                if os.path.isfile(file_path) and filename.endswith(".json"):
                    os.remove(file_path)

    def store_config(self, obj: ConfigObject) -> ConfigObject:
        """
        Stores the ConfigObject as a JSON file in the specified directory.

        Args:
            obj (ConfigObject): The configuration object to store.

        Returns:
            ConfigObject: The stored configuration object.
        """
        filename = f"{obj.type}_{obj.name}.json"
        filepath = os.path.join(self.directory, filename)

        data = {
            "instance": obj.instance,
            "metadata": obj.metadata,
            "created": obj.created if obj.created else int(time.time())  # Use current time if not provided
        }

        with open(filepath, 'w', 'utf-8') as f:
            json.dump(data, f, indent=4)

        return obj

    def get_config(self, object_type: str, object_name: str) -> Optional[ConfigObject]:
        """
        Retrieves the ConfigObject by type and name.

        Args:
            object_type (str): The type of the configuration object.
            object_name (str): The name of the configuration object.

        Returns:
            Optional[ConfigObject]: The retrieved configuration object or None if not found.
        """
        filename = f"{object_type}_{object_name}.json"
        filepath = os.path.join(self.directory, filename)

        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:                
                data = json.load(f)
                return ConfigObject(
                    type=object_type,
                    name=object_name,
                    instance=data['instance'],
                    metadata=data['metadata'],
                    created=data['created']
                )
        return None

    def get_entities(self, object_type: str) -> List[tuple[str, int]]:
        """
        Retrieves all object names of the specified type from the directory.

        Args:
            object_type (str): The type of the objects to retrieve.

        Returns:
            List[tuple[str, int]]: A list of tuples containing the object names and their created timestamps.
        """
        entities = []
        for filename in os.listdir(self.directory):
            if filename.startswith(f"{object_type}_") and filename.endswith(".json"):
                object_name = filename[len(object_type) + 1:-5]  # Remove type and ".json"
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    entities.append((object_name, data['created']))
        return entities
