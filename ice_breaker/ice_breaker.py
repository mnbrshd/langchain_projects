from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import linkedin_lookup_agent
from output_parsers import summary_parser, Summary
from dotenv import load_dotenv
from typing import Tuple

def ice_break_with(name: str, company: str) -> Tuple[Summary, str]:
        linkedin_username = linkedin_lookup_agent(name=name, company=company)
        linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_username, mock=False)

        summary_template = """
            Given the LinkedIn information {information} about a person, I want you to create:
            1. a short summary
            2. two interesting facts about them
            \n{format_instructions}
        """

        summary_prompt_template = PromptTemplate(
            input_variables=["information"], template=summary_template, partial_variables={"format_instructions": summary_parser.get_format_instructions()})

        llm = Ollama(model="llama3")

        chain = summary_prompt_template | llm | summary_parser

        res: Summary = chain.invoke(input={"information": linkedin_data})

        return res, linkedin_data.get("profile_pic_url")



if __name__ == "__main__":
    load_dotenv()
    ice_break_with("Hassan Moin", "siParadigm")