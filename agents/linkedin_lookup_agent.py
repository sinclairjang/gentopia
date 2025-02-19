from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from tools.tools import get_profile_url_tavily, scrape_linkedin_profile

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

    # Let the agent handle the entire lookup process
    formatted_input = f"Find and extract details from the LinkedIn profile of {name}."

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

    return result["output"]

if __name__ == "__main__":
    profile_info = lookup("AI 스토리텔러 김우정")
    print(profile_info)
