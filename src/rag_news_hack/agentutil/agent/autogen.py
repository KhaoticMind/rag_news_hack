
from typing import Annotated, Dict, List, TypedDict

from autogen import (ConversableAgent, GroupChat, GroupChatManager,
                     UserProxyAgent)

from ..ragstore import RAGDatabase
from ..tool import BasicTool
from .base import ToolAgent


class AutogenConfig(TypedDict):
    name: Annotated[str, 'The name we will use to identify the agent']
    system_prompt: Annotated[str, 'The system prompt this agent will use']
    description: Annotated[str,
                           'A description for this agent. Used as introduction on group chats']
    model: Annotated[str, 'The name of the model we are going to call']
    api_type: Annotated[str,
                        'The type of the API to use, for example: openai, anthropic']
    additional_llm_config: Annotated[dict,
                                     'Additional parameters to be passed to the agent llm config such as temperatura']


class AutogenBasicAgent(ToolAgent):
    def __init__(self,  agent_config: AutogenConfig, max_rounds: int = 10, rag_databases: list[RAGDatabase] = [], tools: list[BasicTool] = [], additional_agents: list[AutogenConfig] = [], chat_manager: AutogenConfig = None):
        super().__init__(rag_databases=rag_databases, tools=tools)

        self.user_proxy = UserProxyAgent(
            name='user_proxy',
            code_execution_config=False,
            human_input_mode='NEVER',
            llm_config=None,
            is_termination_msg=lambda msg: 'content' in msg and msg[
                'content'] and "TERMINATE" in msg["content"],
        )

        self.llm_agent = ConversableAgent(
            name=agent_config.get('name', 'assistant'),
            system_message=agent_config.get('system_prompt', None),
            code_execution_config=False,
            description=agent_config.get('description', None),
            human_input_mode="NEVER",
            llm_config={'model': agent_config['model'],
                        'api_type': agent_config['api_type']},
        )

        for name, tool in self.tools.items():
            self.llm_agent.register_for_llm(
                name=name, description=tool.get_description())(tool)
            self.user_proxy.register_for_execution(name=name)(tool)

    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict:

        chat_input = self._retrieve_rag_information(messages)

        # Send the history of the chat to the agent, so he knows what we are talking about
        self.user_proxy.clear_history(self.llm_agent)
        self.llm_agent.clear_history(self.user_proxy)

        for m in chat_input:
            if m['role'] == 'user':
                self.user_proxy.send(
                    message=m, recipient=self.llm_agent, request_reply=False, silent=True)
            else:
                self.llm_agent.send(
                    message=m, recipient=self.user_proxy, request_reply=False, silent=True)

        res = self.user_proxy.initiate_chat(
            recipient=self.llm_agent,
            clear_history=False,
            message=chat_input[-1]['content'],
            max_turns=10
        )

        # print(f'{res=}')
        if res:
            answer = res.chat_history[-1]['content'].removesuffix(
                'TERMINATE').strip()
        else:
            answer = 'SEM RESPOSTA'

        return self._generate_completion_response(message_content=answer, finish_reason='stop', model='modelo', usage={"prompt_tokens": 0,
                                                                                                                       "completion_tokens": 0,
                                                                                                                       "total_tokens": 0})
