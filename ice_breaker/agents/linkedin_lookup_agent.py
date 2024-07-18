from langchain import hub
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain_community.tools import Tool
from tools.tools import get_profile_url_tavily
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()


def linkedin_lookup_agent(name: str, company: str) -> str:

    llm = Ollama(model="llama3")

    template = """Given the full name {name_of_person} who works at {company}, I want you to give me a link to their LinkedIn profile page. Your answer should only contain a URL."""

    prompt_template = PromptTemplate(template=template, input_variables=["name_of_person"])

    tools_for_agent = [
        Tool(
            name="Crawl Google for LinkedIn Profile URL",
            func=get_profile_url_tavily,
            description="useful for when you need to get a LinkedIn Page URL."
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name, company=company)}
    )
    
    linkedin_profile_url = result["output"]

    return linkedin_profile_url