from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from schemas import Analyst, InterviewState, SearchQuestion
from langchain_core.messages import SystemMessage, get_buffer_string
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic.
Your goal is to boil down to interesting and specific insights retlated to your topic.

1. Interesting: Insights that peopel will find surpising or non-obvious.
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic to focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona and then ask your question.
If you have already introduced yourself, do not do it again. Just ask your question.
Continue to ask questions to drill eown and refine your understanding of the topic.
Make sure to only ask one question at a time.
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you.
"""

SEARCH_INSTRUCTIONS = """You will be given a conversation between an anaylyst and an expert.
Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.

First, analyze the full conversation.
Pay particular attention to the final question posed by the analyst.
Convert this final question into a well-structured web search query.
"""

ANSWER_INSTRUCTIONS = """You are an expert being interviewed by an analyst.
Here is analyst area of focus: {goals}.

Your goal is to answer a question posed by the interviewer.

To answer question, use this context:
{context}

When answering questions, follow these guidelines:

1. Use only the information provided in the context.
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
3. The context contain sources at the topic of each individual document.
4. Include these sources of your answer next to any relevant statements. For example , for source #1 use [1].
5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc.
6. If the source is <Document source="assistant/docs/llama3_1.pdf" page="7"> then just list
[1] assistant/docs/llama3_1.pdf, page 7

And skip the addition of the brackets as well as the Document source preamble in your citation.
"""

SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer.
Your task is to create a short, easily digestible summary of a report based on a set of source documnets.

1. Analyze the content of the source documents.
- The name of the source document is at the start of the document with the <Document> tag

2. Create a report structure using markdown formatting
- Use ## for the section title
- Use ### for the subsection title

3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging base upon the focus area of the analyst:
{focus}
5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst.
- Emphasize what is novel, interesting or surprising about insights gathered from the interview.
- Create a numbered list of source documents as you use them
- Do not mention the name of there interviewers or experts
- Aim for approximately 400 works maximum
- Use numbered sources in your report (e.g. [1], [2]) based on information from source documents

6. In the sources section:
- Include all sources used in your report.
- Provide full links of relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:
[3] https://ai.meta.com/blog/meta-llama-3-1/

8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed
"""

MAX_RESULTS = 3

class InterviewAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.graph = self._build_graph()
        self.search = TavilySearch(max_results=MAX_RESULTS)

    def _build_graph(self):
        def ask_question(state: InterviewState):
            """Node to generate a question"""
            analyst = state.analyst
            messages = state.messages
            system_message = SystemMessage(content=QUESTION_INSTRUCTIONS.format(goals=analyst.persona))
            question = self.llm.invoke([system_message, *messages])
            return {"messages": [question]}

        def search_web(state: InterviewState):
            """Retrieve docs from web search"""

            structured_llm = self.llm.with_structured_output(SearchQuestion)
            search_system_message = SystemMessage(content=SEARCH_INSTRUCTIONS)
            search_query = structured_llm.invoke([search_system_message, *state.messages])
            data = self.search.invoke({"query": search_query.search_query})
            if isinstance(data, dict):
                search_docs = data.get("results", [])
            else:
                search_docs = []

            formatted_docs = "\n\n---\n\n".join([
                f'<Document href="{doc.get("url")}"/>\n{doc.get("content")}\n</Document>'
                for doc in search_docs
            ])
            return {"context": [formatted_docs]}

        def search_wikipedia(state: InterviewState):
            """Retrieve data from Wikipedia"""

            structured_llm = self.llm.with_structured_output(SearchQuestion)
            search_system_message = SystemMessage(content=SEARCH_INSTRUCTIONS)
            search_query = structured_llm.invoke([search_system_message, *state.messages])
            search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=MAX_RESULTS).load()
            formatted_docs = "\n\n---\n\n".join([
                f'<Document source="{doc.metadata.get("source")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ])
            return {"context": [formatted_docs]}

        def generate_answer(state: InterviewState):
            """Node to answer the question"""
            analyst = state.analyst
            messages = state.messages
            context = state.context
            system_message = SystemMessage(content=ANSWER_INSTRUCTIONS.format(goals=analyst.persona, context=context))
            answer = self.llm.invoke([system_message, *messages])
            answer.name = "expert"
            return {"messages": [answer]}

        def save_interview(state: InterviewState):
            """Save the interview"""
            full_interview = get_buffer_string(state.messages)
            return {"interview": full_interview}

        def route_message(state: InterviewState, name: str = "expert"):
            """Route between question and answer nodes"""

            messages = state.messages
            max_num_turns = state.max_num_turns
            num_responses = len(
                [m for m in messages if isinstance(m, AIMessage) and m.name == name]
            )

            if num_responses >= max_num_turns:
                return "save_interview"

            last_question = messages[-2]

            if "Thank you so much for your help!" in last_question.content:
                return "save_interview"
            return "ask_question"

        def write_section(state: InterviewState):
            """Write a section of the report"""
            context = state.context
            analyst = state.analyst
            system_message = SystemMessage(content=SECTION_WRITER_INSTRUCTIONS.format(focus=analyst.description))
            human_msg = HumanMessage(content=f"Us this source to write your section: {context}")
            section = self.llm.invoke([system_message, human_msg])
            return {"sections": [section.content]}

        def build_interview_section_graph():
            """Build the graph for writing a section of the report"""
            builder = StateGraph(InterviewState)
            builder.add_node("ask_question", ask_question)
            builder.add_node("search_web", search_web)
            builder.add_node("search_wikipedia", search_wikipedia)
            builder.add_node("generate_answer", generate_answer)
            builder.add_node("save_interview", save_interview)
            builder.add_node("write_section", write_section)

            builder.add_edge(START, "ask_question")
            builder.add_edge("ask_question", "search_web")
            builder.add_edge("ask_question", "search_wikipedia")
            builder.add_edge(["search_web", "search_wikipedia"], "generate_answer")
            builder.add_conditional_edges("generate_answer", route_message, ["ask_question", "save_interview", "write_section"])
            builder.add_edge("save_interview", "write_section")
            builder.add_edge("write_section", END)
            memory = MemorySaver()
            return builder.compile(checkpointer=memory)
        return build_interview_section_graph()