from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from linkedin import get_linkedin_info
# from langchain.chat_models import ollama
from linkedin_lookup_agent import lookup


def get_info_of(name: str):

    linkedin_url = lookup(name)
    information = get_linkedin_info(linkedin_url)

    summary_template = """
    Given the linkedin information {information} about a person I want create:
    1. Short Summary
    2. Two intresting points    
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    chain = summary_prompt_template | llm
    output = chain.invoke(input={"information":information})

    print("Response:")
    return (output.to_json()['kwargs']['content'])



if __name__ == "__main__":
    
    load_dotenv()

    information = get_info_of("Aadesh Dalvi IBM")
    print(information)

    
