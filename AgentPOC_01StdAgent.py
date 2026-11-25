# AgentPOC_01StdAgent.py

# part-1: Setup & Secrets: brings tools we need
# ==============================================================
# genai to talk to Gemeini
# psycopg to talk to PostgreSQL
# load_dotenv to load secret passwords from a hidden .env file 

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import psycopg

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")


# part-2: Giving the AI "Hands" (the Tools)
# ==============================================================
# Tool-1: get_table_schema -> looking at the blueprint
# Tool-2: get_row_count -> Counting the data

# --- 1. Real PostgreSQL Read-Only Tools ---
# what happens here: opens a connection to access PGS database/schema
def get_table_schema(table_name: str, schema_name: str = "public") -> str:
    """Returns the column names, data types, and nullability for a PostgreSQL table.

    Args:
        table_name: Name of the PostgreSQL table (e.g., 'orders', 'customers').
        schema_name: Database schema name (defaults to 'public').
    """
    query = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
    """
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (schema_name, table_name.lower()))
                rows = cur.fetchall()

        if not rows:
            return f"Table '{schema_name}.{table_name}' was not found."

        columns = [
            f"{col} ({dtype}, nullable={nullable})"
            for col, dtype, nullable in rows
        ]
        return f"Schema for {schema_name}.{table_name}: " + ", ".join(columns)

    except Exception as e:
        return f"Database error inspecting schema: {e}"

# Tool-2: what happens here -> Read coutns 

def get_row_count(
    table_name: str, where_clause: str = "", schema_name: str = "public"
) -> str:
    """Returns the exact row count for a PostgreSQL table with an optional WHERE filter.

    Args:
        table_name: Name of the PostgreSQL table.
        where_clause: Optional SQL WHERE condition without the word 'WHERE'
          (e.g., "status = 'FAILED'").
        schema_name: Database schema name (defaults to 'public').
    """
    # Use psycopg.sql to safely quote schema and table identifiers
    from psycopg import sql

    try:
        with psycopg.connect(DB_URL) as conn:
            # Force read-only transaction for safety
            conn.read_only = True
            with conn.cursor() as cur:
                base_query = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name.lower()),
                )
                if where_clause.strip():
                    # Note: In production, validate or restrict where_clause to prevent SQL injection
                    full_query = sql.SQL("{} WHERE {}").format(
                        base_query, sql.SQL(where_clause)
                    )
                else:
                    full_query = base_query

                cur.execute(full_query)
                count = cur.fetchone()[0]

        return (
            f"Row count for {schema_name}.{table_name}"
            f" ({where_clause or 'all rows'}): {count:,}"
        )

    except Exception as e:
        return f"Database error counting rows: {e}"

# Part-3: Given the Agent a "Brain"  (Gemini)
# =====================================================================================

# --- 2. Run the Standard Agent Loop ---
def main():
    # connecting the brain (Gemini) to the ahnds inside main()
    # Startups the Google Gemini Client - automatically finds  GEMINI_API_KEY from .env.

    client = genai.Client()

    # system_instruction -> agent's job description. You are telling to agent what to do. 
    # tools = -> Telling to agent about the tools (two python functions) we created 
    # thinking_config --> telling to agent how fast it has to think, LOW or MEDIUM or FAST
    # automatic_function_calling --> a safety circuit breaker.  \ 
    #         if Gemini gets confused and keeps calling tools in a loop, then python will stop it after 4 tries so you don't waste money. 
    config = types.GenerateContentConfig(
        system_instruction=(
            "You are a PostgreSQL DataOps assistant. Always inspect the table"
            " schema first to verify exact column names before running row"
            " counts."
        ),
        tools=[get_table_schema, get_row_count],
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW
        ),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            maximum_remote_calls=4
        ),
    )

    # Session - creates a chat session using Gemini with all rules and tools attached 'config'
    chat = client.chats.create(
        model="gemini-3.5-flash-lite",
        config=config,
    )

    # Part-4: What happens Behind the Scenes when you run it
    prompt = (
        "Check the schema for the 'orders' table, and tell me how many PENDING"
        " orders we have."
    )
    response = chat.send_message(prompt)
    print(f"Agent Response:\n{response.text}")


if __name__ == "__main__":
    main()
