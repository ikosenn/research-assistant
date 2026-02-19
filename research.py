from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from schemas import ResearchState
from interview import InterviewAgent
from langgraph.types import Send
from schemas import ResearchState

REPORT_WRITER_INSTRUCTIONS = """You are a technical writer creating a report on this overall topic:
{topic}

You have a team of analysts. Each analyst has done two things:

1. They conducted an interview with an expert on a specific sub-topic.
2. The write up their finding into a memo.

Your task:
1. You will be given a collection of memos from your analysts
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp, overall summary that ties together the central ideas from all the memos.
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:

1. Use markdown formatting.
2. Include no pre-amble for the report.
3. Use no sub-heading.
4. Start your report with a single title header: # Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, with will be annotated in brackes, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat. Make sure to include the source number in the list.
For example:
[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from:

{context}
"""

INTRO_CONCLUSION_INSTRUCTIONS = """You are a technical writer finishing a report on a {topic}
You will be given all of the sections of the report.

Your job is to write a crisp and compelling introduction or conclusion section.

The user will intruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting.

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header.

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {context}
"""

class ResearchAgent:
    def __init__(self, llm: ChatOpenAI, max_num_turns: int = 5):
        self.llm = llm
        self.graph = self._build_graph()
        self.max_num_turns = max_num_turns

    def _build_graph(self):
        def initiate_interview(state: ResearchState):
            """Start the interview"""
            interviews = []
            for analyst in state.analysts:
                topic = state.topic
                interviews.append(Send("conduct_interview", {
                    "analyst": analyst,
                    "max_num_turns": self.max_num_turns,
                    "messages": [HumanMessage(content=f"So you said you were wring an article on {topic}?")]
                }))
            if len(interviews) == 0:
                return END
            return interviews

        def write_report(state: ResearchState):
            """Write the report"""
            sections = state.sections
            topic = state.topic
            formatted_str_sections = "\n\n".join([section for section in sections])
            system_message = SystemMessage(content=REPORT_WRITER_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections))
            human_msg = HumanMessage(content="Write a report based upon these memos.")
            report = self.llm.invoke([system_message, human_msg])
            return {"content": report.content}

        def write_introduction(state: ResearchState):
            """Write the introduction"""
            sections = state.sections
            topic = state.topic
            formatted_str_sections = "\n\n".join([section for section in sections])
            system_message = SystemMessage(content=INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections))
            human_msg = HumanMessage(content="Write an introduction based upon these sections.")
            introduction = self.llm.invoke([system_message, human_msg])
            return {"introduction": introduction.content}

        def write_conclusion(state: ResearchState):
            """Write the conclusion"""
            sections = state.sections
            topic = state.topic
            formatted_str_sections = "\n\n".join([section for section in sections])
            system_message = SystemMessage(content=INTRO_CONCLUSION_INSTRUCTIONS.format(topic=topic, context=formatted_str_sections))
            human_msg = HumanMessage(content="Write a conclusion based upon these sections.")
            conclusion = self.llm.invoke([system_message, human_msg])
            return {"conclusion": conclusion.content}

        def finalize_report(state: ResearchState):
            """Finalize the report"""
            content = state.content
            if content.startswith("## Insights"):
                content = content.strip("## Insights")
            if "## Sources" in content:
                try:
                    content, sources = content.split("\n## Sources\n")
                except:
                    sources = None
            else:
                sources = None

            final_report = state.introduction + "\n\n" + content + "\n\n" + state.conclusion
            if sources is not None:
                final_report += "\n\n## Sources\n" + sources
            return {"final_report": final_report}

        def build_researcher_graph():
            """Build the research graph"""
            builder = StateGraph(ResearchState)
            interview_agent = InterviewAgent(self.llm)
            builder.add_node("conduct_interview", interview_agent.graph)
            builder.add_node("write_report", write_report)
            builder.add_node("write_introduction", write_introduction)
            builder.add_node("write_conclusion", write_conclusion)
            builder.add_node("finalize_report", finalize_report)

            builder.add_conditional_edges(START, initiate_interview, ["conduct_interview", END])
            builder.add_edge("conduct_interview", "write_report")
            builder.add_edge("conduct_interview", "write_introduction")
            builder.add_edge("conduct_interview", "write_conclusion")
            builder.add_edge(["write_report", "write_introduction", "write_conclusion"], "finalize_report")
            builder.add_edge("finalize_report", END)
            memory = MemorySaver()
            return builder.compile(checkpointer=memory)
        return build_researcher_graph()