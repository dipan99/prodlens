from logger import Logging
import psycopg2
import os
from dotenv import load_dotenv
from time import time

Logging.setLevel()
load_dotenv()

# Database configuration
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'electronics_db'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT', '5432')
}

def sql_query(sql: str) -> str:
    try:
        Logging.logDebug(f"Retrieving DB output for the query: {sql}")

        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        Logging.logDebug(f"Connected to: {version[0]}")
        
        # List tables
        start = time()
        cursor.execute(sql)
        Logging.logInfo(f"Time taken to fetch response: {round(time() - start, 2)} seconds.")
        tables = cursor.fetchall()

        cursor.close()
        conn.close()
        Logging.logDebug(f"Retrieved tables:\n{tables}\n")
        return tables        
    except Exception as e:
        Logging.logError(str(e))
        raise e


if __name__ == "__main__":
    prompt = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """

    tables = sql_query(prompt)

    print(f"\nTables in database:")
    for table in tables:
        print(f"  - {table[0]}")