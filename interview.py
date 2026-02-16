from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from schemas import Analyst, InterviewState
from langchain_core.messages import SystemMessage

QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specifi topic.
Your goal is to boil down to interesting and specific insights retlated to your topic.

1. Interesting: Insights that peopel will find surpising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic to focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona and then ask your question.
Continue to ask questions to drill eown and refine your understanding of the topic.
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""


class InterviewAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        def generate_question(state: InterviewState):
            """Node to generate a question"""
            analyst = state.analyst
            messages = state.messages
            system_message = SystemMessage(content=QUESTION_INSTRUCTIONS.format(goals=analyst.persona))
            question = self.llm.invoke([system_message, *messages])
            return {"question": question}
