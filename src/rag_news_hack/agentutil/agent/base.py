import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List

from ..ragstore import RAGDatabase
from ..tool.base import BasicTool


class SimpleAgent(ABC):
    def __init__(self, system_prompt: str = '', rag_databases: list[RAGDatabase] = [], rag_on_all_messages: bool = True):
        self.system_prompt = system_prompt
        self.rag_on_all_messages = rag_on_all_messages
        self.rag_databases = rag_databases if rag_databases else []

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:
        pass

    def answer_question(self, question: str) -> str:
        messages = {'role': 'user', 'content': question}
        answer = self.chat_completion(
            [messages])['choices'][0]['message']['content']
        return answer

    def _generate_completion_response(self, model: str, message_content: str, finish_reason: str, usage: dict) -> dict:
        res = {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": message_content.encode(encoding='utf-8').decode(encoding='utf-8'),
                    },
                    "finish_reason": finish_reason,
                    "logprobs": 0.0
                }
            ],
            "usage": {
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"]
            },
            "system_fingerprint": "fp_agentutil"
        }

        return res

    def _retrieve_rag_information(self, messages: list) -> list[dict]:
        """
        Retrieves information from the RAG databases before making a chat completion call.

        :param messages: A list of dictionaries with keys "role" and "content".
        :return: A modified list of messages including relevant RAG information.
        """
        if not self.rag_databases:
            return messages

        # Filter relevant messages (do not include system messages)
        relevant_messages = [
            m for m in messages if m['role'] in ('user', 'assistant')]
        relevant_messages = relevant_messages if self.rag_on_all_messages else [
            relevant_messages[-1]]
        rag_results = []

        # Query RAG databases for relevant content
        for database in self.rag_databases:
            for message in relevant_messages:
                rag_results.extend(database.query_text(message['content']))

        # If we have results, format and append to messages
        if rag_results:
            formatted_rag_content = "\n\n".join(
                [f"#URL: {r.metadata['url']}\n{r.data}" for r in rag_results])

            messages[-1]['content'] += f"\n\nSOURCES{formatted_rag_content}"
        
        return messages


class ToolAgent(SimpleAgent):
    def __init__(self, system_prompt: str = '', rag_databases: list[RAGDatabase] = [], rag_on_all_messages: bool = True, tools: list[BasicTool] = []):
        super().__init__(system_prompt=system_prompt,
                         rag_databases=rag_databases,
                         rag_on_all_messages=rag_on_all_messages)
        self.tools: dict[str, BasicTool] = {tool.get_name(): tool for tool in tools} if tools else {}
