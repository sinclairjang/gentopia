from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily, scrape_linkedin_profile, scrape_profile_mocked
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class LinkedInProfile(BaseModel):
    name: str = Field(description="The person's full name")
    headline: str = Field(description="Their LinkedIn headline")
    location: str = Field(description="Location of the person")
    experience: list[str] = Field(description="A list of past jobs")
    education: list[str] = Field(description="A list of educational institutions")
    skills: list[str] = Field(description="A list of skills")

summary_parser = PydanticOutputParser(pydantic_object=LinkedInProfile)

def lookup(name: str) -> dict:
    """
    Given a person's full name, find their LinkedIn profile and scrape their details.
    The agent autonomously selects tools to retrieve the necessary information.
    Returns a structured dictionary containing the profile data.
    """
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
    )

    tools_for_agent = [
        Tool(
            name="Find LinkedIn Profile",
            func=get_profile_url_tavily,
            description="Find the LinkedIn profile URL of a person based on their name."
        ),
        Tool(
            name="Scrape LinkedIn Profile",
            func=scrape_linkedin_profile,
            description="Scrape details from a LinkedIn profile URL."
        ),
        Tool(
            name="Scrape Cached LinkedIn Profile",
            func=scrape_profile_mocked,
            description="Scrape details from a LinkedIn profile of 김우정(AI Storyteller)"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt,
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,  # ✅ See each step the agent takes
        return_intermediate_steps=True  # ✅ Capture agent reasoning & tool calls
    )

    formatted_input = (
        f"Find and extract details from the LinkedIn profile of {name} and translate them into Korean. "
        f"Use cached results if possible. "
        f"Return the result in the following structured JSON format: {summary_parser.get_format_instructions()}"
    )
    result = agent_executor.invoke(
        input={"input": formatted_input}
    )

    # ✅ Extract tool usage details
    intermediate_steps = result.get("intermediate_steps", [])
    
    used_tools = []
    for step in intermediate_steps:
        if isinstance(step, tuple) and len(step) > 0:
            action = step[0]  # AgentAction object
            used_tools.append(action.tool)

    # Print which tools were used
    if used_tools:
        print(f"Tools used: {', '.join(set(used_tools))}")
    else:
        print("No tools were used.")

    try:
        structured_output = summary_parser.parse(result["output"])
    except Exception as e:
        print("⚠️ Warning: Invalid JSON output. Returning raw response.")
        structured_output = {"raw_output": result["output"]}

    return structured_output

if __name__ == "__main__":
    profile_info = lookup("구글 창업자")
    print(profile_info)
