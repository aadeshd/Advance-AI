import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
# from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
# from langchain_community.embeddings.bedrock import BedrockEmbeddings

## To Run Ollama

# Open CMD
# ollama pull mistral #Model
# ollama pull nomic-embed-text #Embedding model
# ollama serve

CHROMA_PATH = "../chroma"
DATA_PATH = "../articles"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


## Create Chroma DB at predefine location
def create_db():
    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    ## If want to clear db
    # clear_database()


# document_loader object containing text content of each page in pdf, it also has metadata attached with source(file name) and page number.
# Load documents from Directory using document loader
# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


# Split Loaded docs into smaller chunks
# Each page of a pdf is too big to use, thats why we split it into smaller chunks
def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# Function to return embedding function
# Because we need this embedding function at two seperate location 1) Create DB 2) Query Database
# it is necessary to use same embedding at both places
#  We can use embedding by langchain too REF - https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/
# here we are using nomic-embed-text to generate enbedding using ollama
def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings


# Once we have smaller chunks of data and we can use embedding function to create a vector DB
# Vector - a mathematical representation of text that capture the meaning of words, sentences, or documents 
# in simpler words List of numbers
# One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.
def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    global db
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")


# To create IDs for all the chunks in vector DB
# It adds ID as "Source Name:PageNumber:ChunkNumber"
def calculate_chunk_ids(chunks):
    return chunks


# If want to clear the chroma DB
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


## Querying

def query_rag(query_text: str):
    
    # Get the same embedding function used to store the data
    embedding_function = get_embedding_function()

    # Get DB data
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    # To get k most similar chunks from chroma DB w.r.t the query
    results = db.similarity_search_with_score(query_text, k=5)

    # Add all similar chunks as a context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Add context and query in the defined prompt
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Langchain to call Ollama mistral model 
    # Model is already fetch using ollama
    # Pass the new updated prompt to model
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    # Get fetched chunks
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    # print output from LLM and chunks fetched
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def main():
    query_text = input("Ask question: ")
    query_rag(query_text)

if __name__ == "__main__":
    create_db()
    main()
