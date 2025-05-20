"""
1. User query to agent (Count number of letters in given string)
2. Agent wrapped query in special LLM call of ReAct agent (Reasoning and Action)
2.1. Sent LLM call to GPT 3.5 -> Output had information about "Thought", tool, and tool selection
3. Parse all information 
4. use selected tool to get the job done
"""



from dotenv import load_dotenv
from typing import Union
from langchain.agents import tool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str

load_dotenv("D:\Aadesh\Prep\Langchain Course\.env")


# tool decorator - langchain utility to take this function and create a langchain tool from it
# it plug in - name of function, input received, output returns. Populate it in langchain tool class
# as now, this function is a tool, we can not call this function like normal python function call
# as it is a tool now, we have to use .invoke method to use this tool
@tool
def get_text_length(txt:str)->int:
    """ Returns length of a text by characters"""

    print(f"get_text_length enter with {txt=}")
    txt = txt.strip("'\n").strip('"') # Remove non alphabetic chars if any

    return(len(txt))

def find_tool_by_name (tools: list[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Could not find tool {tool_name}")


if __name__ == "__main__":
    
    # print("Hello world")
    # print(get_text_length(txt="Hello world")) # If we do not add tool decorator
    # print(get_text_length.invoke(input={"txt":"Hello world"}))

    tools = [get_text_length]

# ---------------- Thought (LLM CALL) ------------------------

    # Prompt from langchain hub
    # Prompt from "hwchase17/react"
    template = """
    Assistant is a large language model trained by ZhiPuAI.
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    
    TOOLS:
    ------
    Assistant has access to the following tools:
    {tools}
    To use a tool, please use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    Begin!


    New input: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(
        template=template
    ).partial(
        tools = render_text_description(tools=tools),
        tool_names = ", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    intermediate_step = []

# ---------------- Thought END ------------------------


# ---------------- Parsing ------------------------
    agent = (
        {"input" : lambda x : x["input"], "agent_scratchpad" : lambda x : format_log_to_str(x["agent_scratchpad"])} 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
        )

# ---------------- Parsing End ------------------------


# ---------------- Tool Execution ------------------------
    agent_step : Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the text length of 'Dog' in characters?",
            "agent_scratchpad": intermediate_step
            }
        )
    
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))

        print(f"{observation=}")
        intermediate_step.append((agent_step, str(observation)))

    agent_step : Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the text length of 'Dog' in characters?",
            "agent_scratchpad": intermediate_step
            }
        )
    
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
# ---------------- Tool Execution End ------------------------