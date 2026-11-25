# Ver2: uses - client.chats.create()

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


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
    client = genai.Client()

    # Create a chat session with tools configured
    chat = client.chats.create(
        model="gemini-3.5-flash",
        config=types.GenerateContentConfig(tools=[search]),
    )

    # Use send_message instead of generate_content
    response = chat.send_message("What is the weather in Tokyo?")
    print(response.text)


if __name__ == "__main__":
    main()
