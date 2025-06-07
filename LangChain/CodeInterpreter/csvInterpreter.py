from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent


load_dotenv()


def main():
    print("Start...")
    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="episode_info.csv",
        verbose=True,
    )

    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main()
