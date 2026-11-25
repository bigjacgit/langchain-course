# AgentPOC_04DeepAgent_workflow.py

import os
from typing import TypedDict
from deepagents import CompiledSubAgent, create_deep_agent
from dotenv import load_dotenv
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
import psycopg
from psycopg import sql
from pydantic import BaseModel

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Initialize Gemini 3.5 Flash Lite
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


# =====================================================================
# 1. REUSE OUR STAGE 3B CUSTOM WORKFLOW GRAPH AS A COMPILED SUBAGENT
# =====================================================================
class DBWorkflowState(TypedDict):
    messages: list
    table_name: str
    status_filter: str
    schema_details: str


class QueryTarget(BaseModel):
    table_name: str
    status_filter: str = "FAILED"


def db_inspect_node(state: DBWorkflowState) -> dict:
    """Deterministic node: validates schema and queries real local PostgreSQL."""
    user_msg = state["messages"][-1].text
    target = llm.with_structured_output(QueryTarget).invoke(user_msg)
    table = target.table_name.lower()

    with psycopg.connect(DB_URL) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            # 1. Get schema
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table,),
            )
            cols = cur.fetchall()
            if not cols:
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": f"Table '{table}' not found.",
                    }]
                }

            schema_str = ", ".join([f"{c} ({t})" for c, t in cols])

            # 2. Get breakdown by status and total amount
            q = sql.SQL("""
                SELECT status, COUNT(*) AS row_count, COALESCE(SUM(amount), 0) AS total_amount
                FROM public.{}
                GROUP BY status
                ORDER BY row_count DESC;
            """).format(sql.Identifier(table))
            cur.execute(q)
            breakdown = cur.fetchall()

    summary = (
        f"PostgreSQL Audit for '{table}':\n"
        f"- Schema: {schema_str}\n"
        f"- Status Breakdown: {breakdown}"
    )
    return {"messages": [{"role": "assistant", "content": summary}]}


db_builder = StateGraph(DBWorkflowState)
db_builder.add_node("audit_db", db_inspect_node)
db_builder.add_edge(START, "audit_db")
db_builder.add_edge("audit_db", END)

# Step 1: Wrapping Your Custom Workflow as a Specialist (CompiledSubAgent)
# • What it does: Takes the deterministic LangGraph workflow we built earlier (db_builder.compile()) and packages it as a CompiledSubAgent.
# • Why we do this:
#	1. No wrapper boilerplate: Unlike Stage 3 where you had to manually write @tool def ask_database_specialist(...), CompiledSubAgent automatically plugs into the Deep Agent's built-in delegation tool (task).
#	2. Isolated memory: When postgres-auditor runs, its database queries happen in its own separate context window—only the clean summary comes back to the main agent.

# Wrap our deterministic LangGraph workflow as a Deep Agent CompiledSubAgent
postgres_subagent = CompiledSubAgent(
    name="postgres-auditor",
    description=(
        "Audits PostgreSQL table schemas, row counts, and financial/status"
        " breakdowns."
    ),
    runnable=db_builder.compile(),
)


# Step 2: Giving the Main Agent Direct Tools (@tool)
# • What it does: Gives the main Deep Agent direct access to infrastructure metrics and FastAPI error traces.
# • Difference from Stage 3: In Stage 3, the Supervisor had zero direct tools and delegated everything. In Stage 4, the Deep Agent can hold both its own direct tools (tools=[...]) and subagents (subagents=[...]) at the same time.



# =====================================================================
# 2. DEFINE DOMAIN TOOLS FOR THE DEEP AGENT (Infra & FastAPI Logs)
# =====================================================================
@tool
def fetch_infra_metrics(instance_id: str = "prod-aurora-01") -> str:
    """Fetches detailed database server CPU, memory, and connection pool metrics."""
    return (
        f"Instance '{instance_id}' Telemetry:\n"
        "- Peak CPU: 91% at 14:02 UTC\n"
        "- Active DB Connections: 150/150 (Connection Pool Saturated)\n"
        "- Payment-Worker Containers: 2/4 OOMKilled/Restarting"
    )


@tool
def fetch_fastapi_logs(endpoint: str = "/orders") -> str:
    """Fetches raw FastAPI error stack traces for a specific endpoint."""
    return (
        f"FastAPI Error Trace ({endpoint}):\n"
        "[14:02:11 ERROR] sqlalchemy.exc.TimeoutError: QueuePool limit of size"
        " 20 overflow 10 reached, connection timed out, timeout 30.00\n"
        "[14:02:14 ERROR] HTTP 504 Gateway Timeout returned to client during"
        " checkout."
    )


# Step 3: Assembling the Deep Agent Harness (create_deep_agent)
# This is the core block where all the "deep" superpowers get injected automatically:
# When create_deep_agent(...) runs, it quietly attaches three invisible toolkits to your agent without you writing a single line of tool code:
# 1. The Virtual Filesystem Toolkit (write_file, read_file, ls, edit_file, grep): Built into create_deep_agent by default. It creates a virtual folder inside the agent's state (result["files"]) so the agent can save raw data out of its chat memory.
# 2. The Planning Toolkit (TodoListMiddleware): Gives the agent a write_todos tool so it can write a checklist at the start and update each item (pending → in_progress → completed) as it works.
# 3. The Delegation Toolkit (task tool): Created automatically because you passed subagents=[postgres_subagent].


# =====================================================================
# 3. CREATE THE DEEP AGENT (Planning + Virtual Filesystem + Subagents)
# =====================================================================
deep_agent = create_deep_agent(
    model=llm,
    tools=[fetch_infra_metrics, fetch_fastapi_logs],
    subagents=[postgres_subagent],
    # Enable structured To-Do list tracking alongside built-in filesystem & summarization
    middleware=[TodoListMiddleware()],
    system_prompt=(
        "You are a Principal Site Reliability Engineer conducting a deep"
        " post-incident audit.\nFollow this disciplined workflow:\n1. First,"
        " use `write_todos` to create a step-by-step investigation plan.\n2."
        " Delegate the PostgreSQL audit to the `postgres-auditor` subagent, and"
        " gather infra metrics and FastAPI logs using your tools.\n3. Save raw"
        " evidence into `/evidence/db_audit.txt` and `/evidence/telemetry.txt`"
        " using `write_file` so your context stays clean.\n4. Synthesize all"
        " findings and write a polished Markdown runbook to"
        " `/incident_report.md` using `write_file`.\n5. Mark all todos"
        " completed and return a brief executive summary."
    ),
)

# Step 4: What Happens Inside deep_agent.invoke(...) (The Invisible Relay)
# Here is the multi-step sequence the Deep Agent executes on its own:
# 1. Plans First (write_todos): Following your system_prompt, Gemini first creates a structured to-do list (1. Audit Postgres, 2. Check Infra/API logs, 3. Save evidence files, 4. Write /incident_report.md).
# 2. Gathers Data (task + @tools):
# 	• Calls task(subagent="postgres-auditor", ...) → runs your real PostgreSQL GROUP BY status query on orders.
# 	• Calls fetch_infra_metrics() and fetch_fastapi_logs().
# 3. Offloads Raw Notes (write_file): Saves the raw database output into /evidence/db_audit.txt and logs into /evidence/telemetry.txt so its active chat prompt stays clean and uncluttered.
# 4. Authors the Deliverable (write_file): Reads back its evidence files, connects the dots (e.g., DB connection pool saturated at 150/150 → FastAPI timed out → 3 orders failed), and writes a formatted Markdown runbook to /incident_report.md.
# 5. Closes Out (write_todos + Final Message): Marks all to-dos as completed and returns a short executive summary in the final chat message.


# Step 5: Reading the Results & Virtual Files (result.get("files", {}))
# • Comparison to Stages 1–3: In earlier stages, result only had messages (or your custom State keys).
# • What's new here: Because create_deep_agent maintains an in-memory virtual filesystem, result["files"] contains every file the agent created during its run—letting your Python script inspect or export /incident_report.md right after the run finishes.


# =====================================================================
# 4. RUN THE DEEP AGENT & INSPECT THE VIRTUAL FILESYSTEM
# =====================================================================
def main():
    task = (
        "Run a complete post-incident investigation on the 'orders' table"
        " failures. Audit the database, check infra and FastAPI logs, save"
        " evidence files, and generate /incident_report.md."
    )
    print("🚀 Starting Deep Agent Investigation...\n")

    result = deep_agent.invoke({"messages": [HumanMessage(content=task)]})

    # 1. Print the Executive Summary from the final message
    print("=== EXECUTIVE SUMMARY ===")
    print(result["messages"][-1].text)

     # 2. Inspect the Virtual Files created by the Deep Agent during its run!
    files = result.get("files", {})
    print(f"\n=== VIRTUAL FILES CREATED ({len(files)} files) ===")
    
    for file_path in files:
        print(f" 📄 {file_path}")

    if "/incident_report.md" in files:
        print("\n=== CONTENTS OF /incident_report.md ===")
        # DeepAgent stores file content lines/data in the state's virtual filesystem
        file_entry = files["/incident_report.md"]
        content = (
            "\n".join(file_entry["content"])
            if isinstance(file_entry, dict) and "content" in file_entry
            else str(file_entry)
        )
        print(content)


if __name__ == "__main__":
    main()
