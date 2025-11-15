import psycopg2
import os
from dotenv import load_dotenv

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
        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"Connected to: {version[0]}")
        
        # List tables
        cursor.execute(sql)
        tables = cursor.fetchall()

        cursor.close()
        conn.close()

        return tables        
    except Exception as e:
        print(f"Error: {e}")


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