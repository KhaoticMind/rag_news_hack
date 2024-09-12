from openai import OpenAI
from typing import List, Dict
from ..ragstore import RAGDatabase
from ..tool import BasicTool
from .base import SimpleAgent

from .. import SecretRetriever


class OpenAIAgent(SimpleAgent):
    def __init__(self, system_prompt: str = 'You are an Helpfull AI assistant.', rag_databases: list[RAGDatabase] = [], rag_on_all_messages: bool = True, model_name='gpt-4o-mini',  base_url=None, temperature: float = 0.8):
        super().__init__(system_prompt=system_prompt,
                         rag_databases=rag_databases,
                         rag_on_all_messages=rag_on_all_messages)
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        """
        Interacts with the OpenAI API to get a response and returns it in the format expected by the chat completions API.

        :param messages: A list of dictionaries with keys "role" and "content"
        :return: A dictionary with the structure expected from the chat completions API
        """
        chat_input = [
            {"role": "system", "content": self.system_prompt}
        ] + messages

        chat_input = self._retrieve_rag_information(chat_input)

        if self.base_url:
            client = OpenAI(
                base_url=self.base_url,
                api_key= SecretRetriever.get_secret('OPENAI_API_KEY')
            )
        else:
            client = OpenAI(api_key= SecretRetriever.get_secret('OPENAI_API_KEY'))

        response = client.chat.completions.create(
            model=self.model_name,
            messages=chat_input,
            temperature=self.temperature,
        )

        # Extracting the necessary parts from the response
        message_content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage = response.usage.to_dict()

        # Use the helper function from the base class to generate the completion response
        return self._generate_completion_response(self.model_name, message_content, finish_reason, usage)

    def _convert_function_to_json(self, dynamic_function: BasicTool) -> dict:
        function_info = {
            "name": dynamic_function.name,
            "description": dynamic_function.get_description(),
            "parameters": {
                "type": "object",
                "properties": {
                    param: {"type": param_type}
                    for param, param_type in dynamic_function.get_parameter_types().items()
                },
                "required": list(dynamic_function.get_parameter_types().keys())
            },
            "return_type": {
                "type": dynamic_function.get_return_type()
            }
        }
        return function_info
