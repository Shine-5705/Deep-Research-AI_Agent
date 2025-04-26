import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from typing import Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

class ResearchState(TypedDict):
    query: str
    research_data: List[Dict]
    final_answer: str

def research_agent(state: ResearchState) -> ResearchState:
    query = state["query"]
    search_results = tavily_client.search(query, max_results=5)
    research_data = []
    for result in search_results["results"]:
        research_data.append({
            "title": result["title"],
            "url": result["url"],
            "content": result["content"][:500]
        })
    return {"research_data": research_data}

def answer_drafter_agent(state: ResearchState) -> ResearchState:
    research_data = state["research_data"]
    query = state["query"]

    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at drafting concise and accurate answers. Based on the following research data, provide a clear and informative response to the query: "{query}".

        Research Data:
        {research_data}

        Provide a well-structured answer in 3-5 sentences, citing the sources where relevant.
        """
    )

    research_text = "\n".join([f"- {item['title']}: {item['content']} (Source: {item['url']})" for item in research_data])
    chain = prompt | llm
    response = chain.invoke({"query": query, "research_data": research_text})
    final_answer = response.content

    return {"final_answer": final_answer}

def create_workflow():
    workflow = StateGraph(ResearchState)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("answer_drafter_agent", answer_drafter_agent)
    workflow.add_edge("research_agent", "answer_drafter_agent")
    workflow.add_edge("answer_drafter_agent", END)
    workflow.set_entry_point("research_agent")
    return workflow.compile()

def run_deep_research_system(query: str) -> str:
    app = create_workflow()
    initial_state = {
        "query": query,
        "research_data": [],
        "final_answer": ""
    }
    final_state = app.invoke(initial_state)
    return final_state["final_answer"]
