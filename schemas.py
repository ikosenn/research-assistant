from pydantic import BaseModel, Field
from typing import Annotated
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class Analyst(BaseModel):
    affiliation: str = Field(description="The affiliation of the analyst")
    name: str = Field(description="The name of the analyst")
    role: str = Field(description="The role of the analyst in context of the topic.")
    description: str = Field(description="The description of the analyst focus, concerns and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription {self.description}\n"


class Perspective(BaseModel):
    analysts: list[Analyst] = Field(description="The analysts and their roles and affiliations.")


class GeneratAnalystState(BaseModel):
    topic: str
    max_analysts: int
    human_analyst_feedback: str | None = None
    analysts: list[Analyst] | None = None


class InterviewState(BaseModel):
    max_num_turns: int
    context: Annotated[list, operator.add] = []
    analyst: Analyst
    interview: str = ""
    sections: list = []
    messages: Annotated[list[AnyMessage], add_messages] = []


class SearchQuestion(BaseModel):
    search_query: str = Field(None, description="The search query to use to find relevant documents.")
