#  ver0 - client.models.generate_content

from dotenv import load_dotenv
from google import genai
from google.genai import types

from langsmith import traceable
from langsmith.wrappers import wrap_gemini



load_dotenv()


def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


# Automatically picks up GEMINI_API_KEY from your .env file
#client = genai.Client()

# Wrap the Google client so LangSmith captures all LLM calls
client = wrap_gemini(genai.Client())

def main():
    print("Hello from langchain-course!")
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="What is the weather in Chennai?",
        config=types.GenerateContentConfig(
            tools=[search],  # Automatic function calling is enabled by default
        ),
    )
    print(response.text)


if __name__ == "__main__":
    main()
