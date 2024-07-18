from typing import Any
from dotenv import load_dotenv
from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain.tools import Tool
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent


load_dotenv()


def main():
    instructions = """You are an agent designed to write and execute Python code to answer questions.
    You have access to a Python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer."""

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]

    python_agent = create_react_agent(
        prompt=prompt, llm=ChatOllama(model="codellama", temperature=0), tools=tools
    )

    python_agent_executor = AgentExecutor(
        agent=python_agent, tools=tools, verbose=True)

    csv_agent_executor = create_csv_agent(llm=ChatOllama(
        model="codellama", temperature=0), path="code_interpreter/episode_info.csv", verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, allow_dangerous_code=True)

    ######################## ROUTER GRAND AGENT #########################################

    # def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
    #     return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor.invoke,
            description="""Useful when you need to transform natural language to Python and execute the python code,
            returning the results of the code execution
            DOES NOT ACCEPT CODE AS INPUT"""
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""Useful when you need to answer question over episode_info.csv file,
            takes as input the entire question adn returns the answer after running pandas calculations."""
        )
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt, llm=ChatOllama(model="codellama", temperature=0), tools=tools
    )
    grand_agent_executor = AgentExecutor(
        agent=grand_agent, tools=tools, verbose=True)

    print(grand_agent_executor.invoke({
        "input": """Generate and save 15 qr codes that point to "www.udemy.com/course/langchain" in the current working directory using Python"""
    }))


if __name__ == "__main__":
    main()
