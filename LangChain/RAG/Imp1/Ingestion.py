"""
Take an article
split into chunks
Embed data (convert chunks to vectors)
store data into vector store
Uses Pinecone - Popular vector store
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader # to load data from ddiffernt source (Doc/PDF/PPT/CSV/Whatsapp chat etc. check doc for more)
from langchain_text_splitters import CharacterTextSplitter # takes large text and split into small chunks
from langchain_openai import OpenAIEmbeddings # takes text input -> returns vectors in embedding vector space
from langchain_pinecone import PineconeVectorStore #store/add vector in vector space

load_dotenv("D:\Aadesh\Prep\Langchain Course\.env")

if __name__ == "__main__":
    print("In Ingestion")

    print("Load Data...")
    loader = TextLoader("D:/Aadesh/Prep/Langchain Course/RAG/data.txt", encoding = 'UTF-8')
    document = loader.load()

    print("Split Data ....")
    text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_spliter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    print("Embed data ...")
    embedding = OpenAIEmbeddings(openai_api_type=os.environ.get("OPENAI_API_KEY"))

    print("Ingest Data ...")
    PineconeVectorStore.from_documents(texts, embedding, index_name = os.environ.get("INDEX_NAME"))

    print("Done ...")
