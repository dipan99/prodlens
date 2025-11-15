from dotenv import load_dotenv
from openai import OpenAI
from typing import List
import numpy as np
import chromadb
import os
import re

load_dotenv()

client = OpenAI()

chroma_client = chromadb.PersistentClient(path="vectorDB")
collection = chroma_client.get_or_create_collection(name="electronics")


def text_to_sql(prompt: str) -> str:
    schema_context = open(os.path.join("files", "schema.sql"), "r").read()

    system_prompt = (f"""
        You are an expert Text-to-SQL engine.  
        You MUST always generate SQL queries **based strictly on the provided schema_context**.  
        Do NOT invent tables, columns, or relationships that do not exist.

        IMPORTANT:
        - The **products** table is the CENTRAL table in the schema.  
        - ALL queries must reference the products table first.  
        - Any other table (monitor_specs, keyboard_specs, mouse_specs, professional_ratings, brands) must be JOINED to products **only if needed** to answer the user query.
        - Only join tables that contain attributes relevant to the user request.
        - Fully qualify all columns (e.g., products.product_name).
        - Use table aliases (p, b, ms, ks, mos, pr) but keep them readable.
        - Return **only SQL**, without explanation, unless explicitly asked.

        If the user asks for something belonging to:
        - **Monitor** specs → join **monitor_specs** ON monitor_specs.product_id = products.product_id
        - **Keyboard** specs → join **keyboard_specs**
        - **Mouse** specs → join **mouse_specs**
        - **Brand** details → join **brands**
        - **Ratings** → join **professional_ratings**

        ### SCHEMA CONTEXT
        {schema_context}
        Always output the SQL inside triple backticks:
        ```sql
        SELECT ...
    """
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=800
    )
    assistant_text = resp.choices[0].message.content.strip()
    sql = re.sub(r"```(?:sql)?\s*([\s\S]*?)```", r"\1", assistant_text)
    return sql


def get_embeddings(texts: List[str], model="text-embedding-3-small") -> List[np.array]:
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [d.embedding for d in response.data]


def rag_query(prompt: str) -> dict:
    query_embedding = client.embeddings.create(
        input=[prompt],
        model="text-embedding-3-small"
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    return results