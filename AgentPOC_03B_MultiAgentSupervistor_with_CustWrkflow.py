# AgentPOC_03B_MultiAgentSupervistor_with_CustWrkflow.py

import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
import psycopg
from psycopg import sql
from pydantic import BaseModel

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


# =====================================================================
# 1. SPECIALIST #1: CUSTOM WORKFLOW AGENT (LangGraph StateGraph)
#    Enforces: Extract Target -> Validate Schema -> Count Rows -> Report
# =====================================================================
class DBWorkflowState(TypedDict):
    task: str
    table_name: str
    status_filter: str
    schema_valid: bool
    schema_details: str
    report: str


class QueryTarget(BaseModel):
    table_name: str
    status_filter: str = ""


def db_node_1_validate_schema(state: DBWorkflowState) -> dict:
    """Node 1: Extract table/filter and verify the table exists in PostgreSQL."""
    extractor = llm.with_structured_output(QueryTarget)
    target = extractor.invoke(state["task"])
    table = target.table_name.lower()

    query = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position;
    """
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (table,))
            rows = cur.fetchall()

    if not rows:
        return {
            "table_name": table,
            "status_filter": target.status_filter,
            "schema_valid": False,
            "report": (
                f"Validation Failed: Table '{table}' does not exist in public"
                " schema."
            ),
        }

    schema_str = ", ".join([f"{col} ({dtype})" for col, dtype in rows])
    return {
        "table_name": table,
        "status_filter": target.status_filter,
        "schema_valid": True,
        "schema_details": schema_str,
    }


def db_node_2_count_rows(state: DBWorkflowState) -> dict:
    """Node 2: Run safe read-only row count ONLY if schema validation passed."""
    with psycopg.connect(DB_URL) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            if state["status_filter"].strip():
                q = sql.SQL(
                    "SELECT COUNT(*) FROM public.{} WHERE status = %s"
                ).format(sql.Identifier(state["table_name"]))
                cur.execute(q, (state["status_filter"].strip(),))
            else:
                q = sql.SQL("SELECT COUNT(*) FROM public.{}").format(
                    sql.Identifier(state["table_name"])
                )
                cur.execute(q)
            count = cur.fetchone()[0]

    return {
        "report": (
            f"Table '{state['table_name']}' verified"
            f" ({state['schema_details']}). Exact row count for filter"
            f" '{state['status_filter'] or 'ALL'}': {count}."
        )
    }


def route_db_validation(state: DBWorkflowState) -> str:
    return "count_rows" if state.get("schema_valid") else END


# Build & compile the Sub-Agent's StateGraph
db_graph = StateGraph(DBWorkflowState)
db_graph.add_node("validate_schema", db_node_1_validate_schema)
db_graph.add_node("count_rows", db_node_2_count_rows)

db_graph.add_edge(START, "validate_schema")
db_graph.add_conditional_edges(
    "validate_schema",
    route_db_validation,
    {"count_rows": "count_rows", END: END},
)
db_graph.add_edge("count_rows", END)

# Compiled LangGraph Workflow acts as Specialist #1!
postgres_workflow_specialist = db_graph.compile()


# =====================================================================
# 2. SPECIALIST #2 & #3: STANDARD AGENTS (create_agent)
# =====================================================================
@tool
def check_db_instance_health(instance_id: str = "prod-aurora-01") -> str:
    """Checks CPU, memory, and active connection counts on the database server."""
    return (
        f"Instance '{instance_id}': CPU=88% (High), Memory=72%, Active"
        " Connections=145/150."
    )


infra_specialist = create_agent(
    model=llm,
    tools=[check_db_instance_health],
    system_prompt=(
        "You are a Cloud Infrastructure Specialist. Diagnose server CPU and"
        " connections."
    ),
)


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
        " errors."
    ),
)


# =====================================================================
# 3. WRAP ALL SPECIALISTS AS TOOLS FOR THE SUPERVISOR
# =====================================================================
@tool
def ask_database_specialist(task: str) -> str:
    """Delegate any PostgreSQL table schema or row count checks to the DB Workflow Specialist."""
    print(f"  [Supervisor -> DB Workflow Graph]: {task}")
    # Notice: Calling .invoke() on a compiled StateGraph works seamlessly!
    res = postgres_workflow_specialist.invoke({"task": task})
    return res["report"]


@tool
def ask_infra_specialist(task: str) -> str:
    """Delegate server CPU or connection health checks to the Infra Specialist."""
    print(f"  [Supervisor -> Infra Agent]: {task}")
    res = infra_specialist.invoke({"messages": [HumanMessage(content=task)]})
    return res["messages"][-1].text


@tool
def ask_api_log_specialist(task: str) -> str:
    """Delegate FastAPI endpoint error log checks to the API Log Specialist."""
    print(f"  [Supervisor -> API Log Agent]: {task}")
    res = api_log_specialist.invoke({"messages": [HumanMessage(content=task)]})
    return res["messages"][-1].text


# =====================================================================
# 4. SUPERVISOR AGENT (Orchestrates Both Graph & Standard Sub-Agents)
# =====================================================================
supervisor_agent = create_agent(
    model=llm,
    tools=[
        ask_database_specialist,
        ask_infra_specialist,
        ask_api_log_specialist,
    ],
    system_prompt=(
        "You are a Site Reliability Supervisor. Delegate tasks to the right"
        " specialists and synthesize a clear root-cause summary."
    ),
)


def main():
    incident_question = (
        "Users are reporting failed orders! Check how many FAILED rows exist in"
        " the 'orders' table, check if our database server is overloaded, and"
        " check the /orders FastAPI logs."
    )
    print(f"User Request:\n{incident_question}\n")
    result = supervisor_agent.invoke(
        {"messages": [HumanMessage(content=incident_question)]}
    )

    print("\n--- Final Supervisor Root-Cause Report ---")
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
