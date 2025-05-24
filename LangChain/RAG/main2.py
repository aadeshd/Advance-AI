import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough

load_dotenv("D:\Aadesh\Prep\Langchain Course\.env")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    print("Retrieving ...")

    query = "What PCMC official Nikam said about citizen's right to file their objections"

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )


    template = """
    Use following piece of context to answer the question at the end.
    If you don't know the answer, just say that you don't try to make up an answer.
    Use three senteces maximum and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of answer.

    {context}

    Question : {question}

    Helpful Answer:
    
    """

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(query)

    print(result)