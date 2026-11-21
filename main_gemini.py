from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from langchain_core.tracers.langchain import wait_for_all_tracers

tracing_enabled = os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
key_present = bool(os.getenv("LANGSMITH_API_KEY", "").strip())

print(f"LangSmith tracing configured: {'ON' if tracing_enabled else 'OFF'}")
print(f"LangSmith API key: {'present' if key_present else 'missing'}")


def main():
    load_dotenv()
    print("Hello from langchain-course!")

    information = """
    PubScribeLink is a collaborative platform connecting academic researchers
    with professional science writers, editors, and illustrators. It streamlines
    the publication process by matching authors with subject-matter experts who
    refine manuscripts, create impactful visual abstracts, and ensure compliance
    with journal standards, ultimately accelerating the journey from research
    discovery to peer-reviewed publication.
    """

    summary_template = """
    Based on the information below, provide:
    1. A short summary of the platform.
    2. Two interesting facts about the platform.

    Information:
    {information}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    chain = summary_prompt_template | llm

    #response = chain.invoke({"information": information})
    #print(response.text)

    try:
        response = chain.invoke(
            {"information": information},
            config={"run_name": "Gemini - PubScribeLink Summary"},
        )
        print(response.content)
        print(response.text)
    finally:
        wait_for_all_tracers()


# uv run python test_google_api.py
if __name__ == "__main__":
    main()

