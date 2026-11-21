import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Add only models verified as free
# for your Google AI Studio project.
FREE_MODELS = {
    "gemini-3.5-flash-lite",
    "gemini-embedding-001",
}

model = "gemini-3.5-flash-lite"

if model not in FREE_MODELS:
    raise ValueError(
        f"Model {model} is not approved for use."
    )

response = client.models.generate_content(
    model=model,
    contents="Explain what PubScribeLink does in 50 words."
)

print(response.text)