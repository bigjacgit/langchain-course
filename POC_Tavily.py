
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents import create_agent

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from groq import Groq


from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()


def tavily_as_client():
    tavily = TavilyClient()  # Reads TAVILY_API_KEY from .env
    response = tavily.search(query="PostgreSQL 17 new features", max_results=3)

    for item in response["results"]:
        print(item["title"], item["url"])


def tavily_as_langchain():
    # 1. Initialize Groq LLM (fast Llama 3.3 70B model)
    llm = ChatGroq(
        #model="llama-3.3-70b-versatile",
        model="openai/gpt-oss-120b",

        temperature=0,
    )

        # 2. Initialize Tavily Search Tool (returns top 3 LLM-cleaned web results)
    tavily_tool = TavilySearch(max_results=3)

    # 3. Create a standard tool-calling agent
    # Pass directly into tools=[...] without writing a @tool wrapper!

    agent = create_agent(
        model=llm,
        tools=[tavily_tool],
        system_prompt=(
            "You are a helpful DataOps assistant. Always use Tavily search to"
            " verify the latest software versions and release dates before"
            " answering."
        ),
    )

    # 4. Ask the exact same live-web question
    #prompt = (
    #    "What is the latest major stable release version of PostgreSQL, and"
    #    " what are 3 key features introduced in it?"
    #)

    prompt = (
            "who is current president of USA? current prime minister of Inida"
            " what are 3 key great things introduced in it?"
        )

        
    print("🔍 Running Groq + Tavily Search...\n")
    result = agent.invoke({"messages": [HumanMessage(content=prompt)]})

    print("=== GROQ + TAVILY RESPONSE ===")
    print(result["messages"][-1].text)    


if __name__ == "__main__":
    #main()
    #tavily_as_client()
    tavily_as_langchain()