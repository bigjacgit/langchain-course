import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# View models accessible to your project.
for model in client.models.list():
    print(model.name)

# Test text embeddings.
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents="PubScribeLink is an AI publishing platform."
)

vector = result.embeddings[0].values

print("Embedding dimensions:", len(vector))
print("First five values:", vector[:5])
