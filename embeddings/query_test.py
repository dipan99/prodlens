# query_test.py

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import os

# Load environment variables
load_dotenv()


# Connect to Chroma Cloud
client = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant="c8f5973e-54d3-42da-92ad-7f9e48657009",
    database=os.getenv("VECTOR_DB_NAME")
)

collection_name = os.getenv("DB_COLLECTION_NAME")

embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma(
    client=client,
    collection_name=collection_name,
    embedding_function=embeddings,
)

# Ask a test question
query = input("Enter your query: ")

results = vectorstore.similarity_search(query, k=3)

print("\nTop Results:\n" + "="*40)
for i, r in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(r.page_content[:800])
    print("-" * 40)
