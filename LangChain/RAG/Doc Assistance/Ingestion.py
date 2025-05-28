from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

def ingest_docs():
    # Load
    # Split
    # Embed
    # Store
    
    loader = ReadTheDocsLoader(r"langchain-docs\api.python.langchain.com\en\latest", encoding = 'UTF-8')
    raw_docs = loader.load()
    print(f"loaded {len(raw_docs)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    docs = text_splitter.split_documents(raw_docs)

    for doc in docs:
        new_url = doc.metadata['source']
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to load {len(docs)} in Pinecone")

    PineconeVectorStore.from_documents(
        docs,embedding=embeddings, index_name = "langchain-doc-index"
    )

    print("Ingestion Complete")

if __name__ == "__main__":
    ingest_docs()