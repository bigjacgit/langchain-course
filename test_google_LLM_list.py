import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# Add model IDs ONLY after confirming that they
# are free for your project in Google AI Studio.
#
# Example:
# VERIFIED_FREE_MODELS = {"gemini-3.5-flash-lite"}
#
# Keep empty until verified.
VERIFIED_FREE_MODELS = set()


def get_model_type(name, actions):
    name = name.lower()
    actions = set(actions)

    if "embedContent" in actions or "embedding" in name:
        return "EMBEDDING"

    if "tts" in name or "speech" in name or "audio" in name:
        return "VOICE / AUDIO"

    if "imagen" in name or "image" in name:
        return "IMAGE"

    if "video" in name or "veo" in name:
        return "VIDEO"

    if "generateContent" in actions:
        return "TEXT GENERATION"

    return "OTHER"


def list_models():
    print("\nGOOGLE GEMINI MODEL DISCOVERY")
    print("=" * 110)

    print(
        f"{'MODEL NAME':45}"
        f"{'TYPE':20}"
        f"{'FREE STATUS':20}"
        f"{'DISCOVERY'}"
    )

    print("-" * 110)

    free_models = []

    for model in client.models.list():
        name = model.name or ""
        model_id = name.removeprefix("models/")
        actions = model.supported_actions or []

        model_type = get_model_type(name, actions)

        is_free = model_id in VERIFIED_FREE_MODELS

        free_status = (
            "VERIFIED FREE"
            if is_free
            else "NOT VERIFIED"
        )

        print(
            f"{model_id[:44]:45}"
            f"{model_type:20}"
            f"{free_status:20}"
            f"LISTED"
        )

        if is_free:
            free_models.append((model_id, model_type))

    print("\n" + "=" * 70)
    print("VERIFIED FREE MODELS")
    print("=" * 70)

    if not free_models:
        print("No models verified yet.")
        print("Check Google AI Studio pricing and quotas.")
    else:
        for model_id, model_type in free_models:
            print(f"{model_id:45} {model_type}")

    print("\nNOTE:")
    print("LISTED means returned by the Models API.")
    print("It does not guarantee successful inference.")
    print("No paid inference calls were made.")


if __name__ == "__main__":
    list_models()