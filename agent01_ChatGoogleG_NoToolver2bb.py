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
    client = genai.Client()

    # Configure tools AND limit automatic function calling turns
    config = types.GenerateContentConfig(
        tools=[search],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=3  # Prevents infinite tool loops (default is 10)
        ),
    )

    chat = client.chats.create(
        model="gemini-3.5-flash",
        config=config,
    )

    response = chat.send_message("What is the weather in Tokyo?")
    print(f"\nFinal Answer: {response.text}\n")

    # Inspect every step recorded in the chat history
    print("--- Execution Trace ---")
    for i, message in enumerate(chat.get_history(), start=1):
        for part in message.parts:
            if part.text:
                print(f"Step {i} [{message.role} text]: {part.text.strip()}")
            elif part.function_call:
                print(
                    f"Step {i} [{message.role} tool call]:"
                    f" {part.function_call.name}({part.function_call.args})"
                )
            elif part.function_response:
                print(
                    f"Step {i} [{message.role} tool output]:"
                    f" {part.function_response.response}"
                )


if __name__ == "__main__":
    main()
