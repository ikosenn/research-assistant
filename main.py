from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from analyst import AnalystAgent
from schemas import GeneratAnalystState, Analyst

load_dotenv()


def run_analyst_agent(llm: ChatOpenAI) -> list[Analyst] | None:
    topic = input("Enter the topic of the research: ")
    max_analysts = int(input("Enter the number of analysts to generate: "))
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

        for event in graph.stream(generate_analyst_state, thread, stream_mode="values"):
            analysts = event.get('analysts')
            if analysts:
                print("-" * 50)
                for analyst in analysts:
                    print(f"Persona: {analyst.persona}")
                    print("-" * 50)

        state_next = graph.get_state(thread).next
    return graph.get_state(thread).values.get('analysts')




def main():
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    analysts = run_analyst_agent(llm)

if __name__ == "__main__":
    main()
