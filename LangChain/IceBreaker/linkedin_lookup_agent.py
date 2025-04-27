from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
# Create react agent - Build in function received param (LLM, Tool and Prompt) -> returns Agent based on react algorithm
# Agent executor - run time of agent
from langchain import hub
from webcrawler import get_linkedin_profile_tavily


def lookup(name: str) -> str:
    load_dotenv()
    
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    template = """
    Given full name {name_of_person} of a person, I want you to get link to their linked page.
    Your answer should only contain a URL.
    """

    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    tools_for_agent=[
        Tool(
            name = "Crawl Google 4 linkedin profile page",
            func = get_linkedin_profile_tavily,
            description = "useful when you need to get linkedin page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    result = agent_executor.invoke(
        input={"input":prompt_template.format_prompt(name_of_person=name)}
    )

    print("Linkedin profile of the person : ")
    print(result["output"])
    print()
    print("Will fetch info from above URL")
    print()

    # As we are not using scarpein to get linkedin info from url, we are passing a dummy file URL
    print("****As linkedin scarping is not in place, It will get dummy info of person John doe.****")
    return "https://raw.githubusercontent.com/aadeshd/Langchain/refs/heads/main/ScrapinDummy.json"

