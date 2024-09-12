from inspect import Parameter, Signature
from typing import Annotated

from ..agent.base import SimpleAgent
from .base import BasicTool


class AgentTool(BasicTool):
    def __init__(self, agent: SimpleAgent, name: str, description: str, question_template: str = 'Review the text bellow. Its title is {question}\n\n{text}'):
        super().__init__(name=name, description=description)
        self.agent = agent
        self.question_template = question_template

    def __call__(self, **kwargs) -> str:
        question = kwargs['question']
        text = kwargs['text']
        return self.agent.answer_question(question=self.question_template.format(question=question, text=text))

    @classmethod
    def set_signature(cls):
        param1 = Parameter('cls', kind=Parameter.POSITIONAL_OR_KEYWORD)
        param2 = Parameter('question', kind=Parameter.KEYWORD_ONLY,
                           annotation=Annotated[str, 'The objective/title of the text'])
        param3 = Parameter('text', kind=Parameter.KEYWORD_ONLY,
                           annotation=Annotated[str, 'The text you want to be reviewed'])
        params = [param1, param2, param3]
        cls.__call__.__signature__ = Signature(
            parameters=params, return_annotation=Annotated[str, 'The review of the given text'])
