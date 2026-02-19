from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from analyst import AnalystAgent
from schemas import ResearchState, GeneratAnalystState, Analyst
from interview import InterviewAgent
from research import ResearchAgent

load_dotenv()


def run_analyst_agent(llm: ChatOpenAI, topic: str, max_analysts: int) -> tuple[list[Analyst], str]:
    analyst_agent = AnalystAgent(llm)
    generate_analyst_state = GeneratAnalystState(topic=topic, max_analysts=max_analysts)
    thread = {"configurable": {"thread_id": "1"}}
    graph = analyst_agent.graph
    state_next = ("START",)
    while state_next:
        if isinstance(state_next, tuple) and 'human_feedback' in state_next:
            human_feedback = input("Any additional feedback to guide the analyst generation. Press Enter to continue: ")
            human_feedback = human_feedback or None
            graph.update_state(thread, {"human_analyst_feedback": human_feedback}, as_node='human_feedback')
            generate_analyst_state = None

        for event in graph.stream(generate_analyst_state, thread, stream_mode="updates"):
            print("-" * 50)
            print(event)
        print("-" * 50)

        state_next = graph.get_state(thread).next
    return graph.get_state(thread).values.get('analysts'), graph.get_state(thread).values.get('topic')

def conduct_research(llm: ChatOpenAI, analysts: list[Analyst], topic: str, max_num_turns: int) -> str:
    research_state = ResearchState(analysts=analysts, topic=topic)
    research_agent = ResearchAgent(llm, max_num_turns)
    thread = {"configurable": {"thread_id": "1"}}
    graph = research_agent.graph
    for event in graph.stream(research_state, thread, stream_mode="updates"):
        print("-" * 50)
        print(event)
        print("-" * 50)
    return graph.get_state(thread).values.get('final_report')


def main():
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    topic = input("Enter the topic of the research: ")
    max_analysts = int(input("Enter the number of analysts to generate: "))
    max_num_turns = int(input("Enter the maximum number of turns for the interview: "))
    analysts, topic = run_analyst_agent(llm, topic, max_analysts)
    final_report = conduct_research(llm, analysts, topic, max_num_turns)
    print(final_report)

if __name__ == "__main__":
    main()
