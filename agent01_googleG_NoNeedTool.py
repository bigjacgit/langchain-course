from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# Standard Python function (type hints + docstring act as the tool schema)
def search(query: str) -> str:
    """Tool that searches over internet.

    Args:
        query: The query to search for.

    Returns:
        The search result.
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


def main():
    print("Hello from google-genai!")
    # Automatically reads GEMINI_API_KEY or GOOGLE_API_KEY from .env
    client = genai.Client()

    # Pass the Python function directly into tools; Gemini handles the tool loop automatically
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="What is the weather in Tokyo?",
        config=types.GenerateContentConfig(tools=[search]),
    )
    print(response.text)


if __name__ == "__main__":
    main()
