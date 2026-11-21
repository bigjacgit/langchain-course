import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from google import genai


load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = f"""PubScribeLink is a collaborative platform connecting academic researchers with professional science writers, editors, and illustrators. 
            It streamlines the publication process by matching authors with subject-matter experts who refine manuscripts, create impactful visual abstracts, and 
            ensure compliance with journal standards, ultimately accelerating the journey from research discovery to peer-reviewed publication.
        """

    summary_template=f"""
        given the information {information} about a person I want you to create:
        1. short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate{
        input_variables=["information"], template=summary_template
    }

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm
    
    response =chain.invoke(input={"Information": information})
    print(response.content)

if __name__ == "__main__":
    main()
