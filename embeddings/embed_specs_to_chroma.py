from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
from dotenv import load_dotenv
import os
load_dotenv()


# Load PDF
loader = PyPDFLoader("../data/specifications_guide.pdf")
docs = loader.load()

# Split into sections
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))

# Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant="c8f5973e-54d3-42da-92ad-7f9e48657009",
    database=os.getenv("VECTOR_DB_NAME")
)
vectorstore = Chroma(client=client, collection_name=os.getenv("DB_COLLECTION_NAME"), embedding_function=embeddings)

# Add embeddings to Chroma
vectorstore.add_documents(splits)

print("PDF guide successfully embedded to Chroma Cloud!")
