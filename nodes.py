from dotenv import load_dotenv
from logger import Logging
from openai import OpenAI
from typing import List
import numpy as np
import chromadb
import re
import os

load_dotenv()

client = OpenAI()
Logging.setLevel()
chroma_client = chromadb.PersistentClient(path="vectorDB")
collection = chroma_client.get_or_create_collection(name="electronics")


def text_to_sql(prompt: str) -> str:
    try:
        Logging.logDebug(f"Generating SQL query for the prompt: {prompt}")
        schema_context = open("schema.sql", "r").read()

        system_prompt = open(os.path.join("templates", "text2sql.txt")).read()

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content": system_prompt.format(schema=schema_context)},
                {"role":"user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        assistant_text = resp.choices[0].message.content.strip()
        sql = re.sub(r"```(?:sql)?\s*([\s\S]*?)```", r"\1", assistant_text)
        Logging.logDebug(f"SQL Query:\n{sql}\n")
        return sql
    except Exception as e:
        Logging.logError(str(e))
        raise e



def get_embeddings(texts: List[str], model="text-embedding-3-small") -> List[np.array]:
    try:
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        return [d.embedding for d in response.data]
    except Exception as e:
        Logging.logError(str(e))
        raise e 


def rag_query(prompt: str, content_type: str = "spec") -> dict:
    try:
        Logging.logDebug(f"Retrieving RAG context for the prompt: {prompt}")
        query_embedding = client.embeddings.create(
            input=[prompt],
            model="text-embedding-3-small"
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"type": content_type}
        )
        Logging.logDebug(f"RAG Results:\n{results}\n")
        return results
    except Exception as e:
        Logging.logError(str(e))
        raise e
    

if __name__ == "__main__":
    print(text_to_sql("Suggest me some monitors with high refresh rates for gaming."))
    print(rag_query("What is RGB in Monitors", "reviews"))