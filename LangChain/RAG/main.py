import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv("D:\Aadesh\Prep\Langchain Course\.env")

if __name__ == "__main__":
    print("Retrieving ...")

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "What PCMC official Nikam said about citizen's right to file their objections"
    chain = PromptTemplate.from_template(template= query) | llm
    result = chain.invoke(input={})
    # print(result)

    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
    )

    retreieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chains = create_stuff_documents_chain(llm, retreieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chains
    )

    result = retrival_chain.invoke(input={"input": query})

    print(result['answer'])

