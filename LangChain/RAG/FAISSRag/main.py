from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub  
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv("D:\Aadesh\Prep\Langchain Course\.env")

pdf_file_path = "BhauraoPatil.pdf"

if __name__ == "__main__":

    loader = PyPDFLoader(file_path=pdf_file_path)
    documents = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30, separator="\n")
    docs = text_spliter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents=docs, embedding= embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retreieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chains = create_stuff_documents_chain(OpenAI(), retreieval_qa_chat_prompt)

    retrival_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(), combine_docs_chain=combine_docs_chains
    )

    result = retrival_chain.invoke(input={"input": "What was the relation between Gandhi and Bhaurao Patil?"})

    print(result['answer'])
