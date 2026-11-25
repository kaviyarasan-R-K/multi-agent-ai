from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.multi_agent_ai.tools.tools import web_search,scrape_url
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature = 0
)

def build_search_agent():
    return create_agent(
        model = llm,
        tools = [web_search],
        system_prompt= "You are a helpful web search assistant",
    )

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        system_prompt= "You are a helpful web scarpe assistant",
    )

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert writer. Write clear, structured, and insightful reports."),
    ("human", """Write a detailed report based on the information below.
Topic: {topic}

Research gathered:
{research}
"""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert editor and fact-checker. Review the draft report critically "
        "for accuracy against the gathered research, completeness, clarity, and structure. "
        "Point out unsupported claims, missing details, or weak sections with specific, actionable improvements.",
    ),
    (
        "human",
        """Review the draft report below against the original topic and gathered research.

Topic: {topic}

Research gathered:
{research}

Draft report:
{draft}

Provide:
1. Overall verdict (APPROVED or NEEDS REVISION)
2. Factual accuracy check (flag anything not supported by the research)
3. Specific improvements needed for clarity, depth, and structure
""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()


