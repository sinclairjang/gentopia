import os;
from dotenv import load_dotenv;
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from third_party.linkedin import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("Hello LangChain!")

    summary_template = """
        given the information {information} about a person, I want you to create in korean:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template)
    
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini"
    )

    chain = summary_prompt_template | llm
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url="https://www.linkedin.com/in/storyswell/",
        mock=True
    )
    res = chain.invoke(input={"information": linkedin_data})

    print(res)