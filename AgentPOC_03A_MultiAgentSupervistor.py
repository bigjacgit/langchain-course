# AgentPOC_03A_MultiAgentSupervistor.py
# One Supervisor Agent deal with Three Sub agents. 

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg
from psycopg import sql

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Use fast, cost-effective Gemini 3.5 Flash Lite for routing and specialists
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


# =====================================================================
# 1. SPECIALIST #1: POSTGRESQL DATABASE AGENT (Real Local PostgreSQL)
# =====================================================================
@tool
def pg_get_schema(table_name: str) -> str:
    """Returns column names and data types for a PostgreSQL table."""
    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (table_name.lower(),))
                rows = cur.fetchall()
        if not rows:
            return f"Table '{table_name}' not found."
        return ", ".join([f"{col} ({dtype})" for col, dtype in rows])
    except Exception as e:
        return f"DB error: {e}"


@tool
def pg_count_rows(table_name: str, status_filter: str = "") -> str:
    """Counts rows in a PostgreSQL table, optionally filtered by status (e.g., 'FAILED')."""
    try:
        with psycopg.connect(DB_URL) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                if status_filter.strip():
                    q = sql.SQL(
                        "SELECT COUNT(*) FROM public.{} WHERE status = %s"
                    ).format(sql.Identifier(table_name.lower()))
                    cur.execute(q, (status_filter.strip(),))
                else:
                    q = sql.SQL("SELECT COUNT(*) FROM public.{}").format(
                        sql.Identifier(table_name.lower())
                    )
                    cur.execute(q)
                count = cur.fetchone()
        return (
            f"Found {count} rows in '{table_name}' (filter:"
            f" {status_filter or 'NONE'})."
        )
    except Exception as e:
        return f"DB error: {e}"


postgres_specialist = create_agent(
    model=llm,
    tools=[pg_get_schema, pg_count_rows],
    system_prompt=(
        "You are a PostgreSQL Database Specialist. Check schemas and count"
        " rows accurately. Return concise, factual findings."
    ),
)


# =====================================================================
# 2. SPECIALIST #2: CLOUD INFRASTRUCTURE AGENT (Compute & DB Metrics)
# =====================================================================
@tool
def check_db_instance_health(instance_id: str = "prod-aurora-01") -> str:
    """Checks CPU, memory, and active connection counts on the database server."""
    return (
        f"Instance '{instance_id}': CPU=88% (High), Memory=72%, Active"
        " Connections=145/150."
    )


@tool
def check_worker_compute_status() -> str:
    """Checks background worker container health and queue backlog."""
    return (
        "Payment-Worker containers: 2/4 healthy (2 containers restarting due to"
        " timeout)."
    )


infra_specialist = create_agent(
    model=llm,
    tools=[check_db_instance_health, check_worker_compute_status],
    system_prompt=(
        "You are a Cloud Infrastructure Specialist. Diagnose server CPU,"
        " connections, and worker health."
    ),
)


# =====================================================================
# 3. SPECIALIST #3: FASTAPI APPLICATION LOG AGENT
# =====================================================================
@tool
def search_api_errors(endpoint: str = "/orders") -> str:
    """Searches recent FastAPI application error logs for a given endpoint."""
    return (
        f"Recent logs for POST {endpoint}: 3 occurrences of 'HTTP 504 Gateway"
        " Timeout: DB connection pool exhausted'."
    )


api_log_specialist = create_agent(
    model=llm,
    tools=[search_api_errors],
    system_prompt=(
        "You are a FastAPI Application Log Specialist. Summarize recent API"
        " error tracebacks."
    ),
)


# =====================================================================
# 4. WRAP SPECIALISTS AS TOOLS FOR THE SUPERVISOR
# =====================================================================
@tool
def ask_database_specialist(task: str) -> str:
    """Delegate any PostgreSQL table schema or row count questions to the Database Specialist."""
    print(f"  [Supervisor -> DB Specialist]: {task}")
    res = postgres_specialist.invoke({"messages": [HumanMessage(content=task)]})
    return res["messages"][-1].text


@tool
def ask_infra_specialist(task: str) -> str:
    """Delegate server CPU, memory, or worker container health checks to the Infra Specialist."""
    print(f"  [Supervisor -> Infra Specialist]: {task}")
    res = infra_specialist.invoke({"messages": [HumanMessage(content=task)]})
    return res["messages"][-1].text


@tool
def ask_api_log_specialist(task: str) -> str:
    """Delegate FastAPI endpoint error log investigations to the API Log Specialist."""
    print(f"  [Supervisor -> API Log Specialist]: {task}")
    res = api_log_specialist.invoke({"messages": [HumanMessage(content=task)]})
    return res["messages"][-1].text


# =====================================================================
# 5. CREATE & RUN THE SUPERVISOR AGENT
# =====================================================================
supervisor_agent = create_agent(
    model=llm,
    tools=[
        ask_database_specialist,
        ask_infra_specialist,
        ask_api_log_specialist,
    ],
    system_prompt=(
        "You are a Site Reliability Supervisor. You do NOT have direct access"
        " to databases or servers. Break down the user's incident question,"
        " delegate tasks to the right specialist(s), and synthesize a clear"
        " root-cause summary."
    ),
)


def main():
    incident_question = (
        "Users are reporting failed orders! Check how many FAILED rows exist in"
        " the 'orders' table, check if our database server or workers are"
        " overloaded, and check the /orders FastAPI logs to explain what is"
        " happening."
    )
    print(f"User Request:\n{incident_question}\n")
    print("Supervisor Delegating Work...")

    result = supervisor_agent.invoke(
        {"messages": [HumanMessage(content=incident_question)]}
    )

    print("\n--- Final Supervisor Root-Cause Report ---")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
