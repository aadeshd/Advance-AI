from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

information = """
Jyotirao Phule (11 April 1827 – 28 November 1890), also known as Jyotiba Phule, was an Indian social activist, businessman, anti-caste social reformer and writer from Maharashtra.[3][4]

His work extended to many fields, including eradication of untouchability and the caste system and for his efforts in educating women and oppressed caste people.[5] He and his wife, Savitribai Phule, were pioneers of women's education in India.[5][6] Phule started his first school for girls in 1848 in Pune at Tatyasaheb Bhide's residence or Bhidewada.[7] He, along with his followers, formed the Satyashodhak Samaj (Society of Truth Seekers) to attain equal rights for people from lower castes. People from all religions and castes could become a part of this association which worked for the upliftment of the oppressed classes.

Phule is regarded as an important figure in the social reform movement in Maharashtra. The honorific Mahātmā (Sanskrit: "great-souled", "venerable"), was first applied to him in 1888 at a special program honoring him in Mumbai.[8][9][10]
"""

if __name__ == "__main__":
    load_dotenv()

    summary_template = """
    Given the information {information} about a person I want create:
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
    print(output.to_json()['kwargs']['content'])
