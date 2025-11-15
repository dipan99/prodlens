import os
import tempfile
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import chromadb

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
TENANT_ID = os.getenv("TENANT_ID")
DATABASE_NAME = os.getenv("DATABASE_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-large")

app = FastAPI(title="ProdLens Embedding API")

client = chromadb.CloudClient(
    api_key=CHROMA_API_KEY,
    tenant=TENANT_ID,
    database=DATABASE_NAME
)

# Initialize embeddings once
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)


@app.post("/embed")
async def embed_pdf(
    collection_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        # Save file temporarily
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and split PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(400, "PDF contains no extractable text.")

        # Create or connect to the collection
        vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )

        # Add chunks → this auto-embeds them
        vectorstore.add_documents(chunks)

        return {
            "status": "success",
            "collection": collection_name,
            "documents_added": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
