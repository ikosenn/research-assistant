from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from schemas import Analyst, Perspective, GeneratAnalystState

load_dotenv()

ANALYST_INSTRUCTIONS ="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide the creation of the analysts:
{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and / or feedback above

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme.
"""

class AnalystAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        def create_analyst(state: GeneratAnalystState):
            """Create analysts"""
            topic = state.topic
            max_analysts = state.max_analysts
            human_analyst_feedback = state.human_analyst_feedback
            structured_llm = self.llm.with_structured_output(Perspective)
            system_message = SystemMessage(content=ANALYST_INSTRUCTIONS.format(topic=topic, max_analysts=max_analysts, human_analyst_feedback=human_analyst_feedback))
            response = structured_llm.invoke([system_message, HumanMessage(content="Generate the analysts")])
            return {"analysts": response.analysts}

        def human_feedback(state: GeneratAnalystState):
            """No-op node that will be interrupted by a human"""
            pass

        def should_continue(state: GeneratAnalystState):
            """Return the next noded to execute"""

            if state.human_analyst_feedback:
                return "create_analysts"
            return END

        def build_create_analyst_graph():
            """Build the graph for creating analysts"""
            builder = StateGraph(GeneratAnalystState)
            builder.add_node("create_analysts", create_analyst)
            builder.add_node("human_feedback", human_feedback)
            builder.add_edge(START, "create_analysts")
            builder.add_edge("create_analysts", "human_feedback")
            builder.add_conditional_edges(
                "human_feedback",
                should_continue,
                ["create_analysts", END]
            )
            memory = MemorySaver()
            return builder.compile(checkpointer=memory, interrupt_before=['human_feedback'])
        return build_create_analyst_graph()