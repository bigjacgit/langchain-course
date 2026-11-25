# AgentPOC_02Custom Work Flow Agent 
# what make this different from previous one - standard agent
# Point #1 - Control: For StandardAgent, Gemini (brain) decides what tool to run. Here I (human) locked the exact sequence ex. Inspect->human_approval->execute/cancel
# Point #2 - SQL Safety: for StandardAgent, Gemini write raw WHERE strings. Here, LLM only extracts parametes; python uses parameteirzed %s queries ex. WHERE status=%s
# Point #3 - State & Pause: for StandardAgent, stateless single loop, here, uses 'DataOpsState' + 'Memory Server' + 'Interrupt()' to freeze execution mid-fight and resume after huan input 
# Overall - StandardAgent are given tools and said 'figure out what to do.'.  For custom workflow agent, you took freedom and defined a box and said agent to run/perform within that box.  

import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import psycopg
from psycopg import sql

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

# Initialize Gemini 3.5 Flash Lite via LangChain
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


# Step-1: Memory Clipboard (DataOpsState) 
#     For StandardAgent, there was no structured data clipboard. Everything lived inside an unstructured chat history ex. messages list.
#     For here, CustomWrkflowAgent, created a structured data clipboard as memory. Why it matters? Every step in our workflow reads from
#               this clipboard and writes its results back onto it so the next step knows what happened.

# --- 1. Define the Shared Graph State ---
# Every node reads from and writes to this central dictionary
class DataOpsState(TypedDict):
    user_request: str
    table_name: str
    status_filter: str
    affected_rows: int
    approved: bool
    final_message: str


# --- 2. Define the Workflow Nodes (Steps) ---
# Turning "Tools" into "Nodes" like the Assembly Line Stations. 
# For exaample:  parse_and_inspect_node(), human_approval_node(),  execute_delete_node(), cancel_node()

# Station-1: parse_and_inspect_node(AI+Read-only DB)
def parse_and_inspect_node(state: DataOpsState) -> dict:
    """Step 1: Use Gemini to extract the target table/status, then count affected rows in PostgreSQL."""
    print("\n[Step 1] Inspecting request and checking PostgreSQL...")

    # Ask Gemini to extract structured parameters cleanly
    prompt = (
        "Extract the table name and order status to delete from this request: "
        f"'{state['user_request']}'. "
        "Reply ONLY in the format: table_name,status_value (example: orders,FAILED)"
    )
 
    # extraction = llm.invoke(prompt).content.strip()  --> errorred 'content' 
    extraction = llm.invoke(prompt).text.strip()

    table_name, status_filter = [x.strip() for x in extraction.split(",")]

    # Query PostgreSQL to see the exact blast radius BEFORE deleting anything
    with psycopg.connect(DB_URL) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            query = sql.SQL(
                "SELECT COUNT(*) FROM public.{} WHERE status = %s"
            ).format(sql.Identifier(table_name.lower()))

            cur.execute(query, (status_filter,))
            count = cur.fetchone()[0]

    print(
        f"   -> Found {count} rows in '{table_name}' where status ="
        f" '{status_filter}'."
    )
    return {
        "table_name": table_name.lower(),
        "status_filter": status_filter,
        "affected_rows": count,
    }


def human_approval_node(state: DataOpsState) -> dict:
    """Step 2: Pause the workflow and ask a human engineer to approve the DELETE."""
    if state["affected_rows"] == 0:
        print("\n[Step 2] 0 rows matched. Skipping approval.")
        return {"approved": False}

    print("\n[Step 2] ⏸️  WORKFLOW PAUSED FOR HUMAN APPROVAL")

    # interrupt() freezes the graph state and returns this payload to the caller
    decision = interrupt({
        "action": "DELETE",
        "table": state["table_name"],
        "where": f"status = '{state['status_filter']}'",
        "rows_to_delete": state["affected_rows"],
        "question": (
            f"Approve deleting {state['affected_rows']} rows from"
            f" '{state['table_name']}'? (yes/no)"
        ),
    })

    is_approved = str(decision).strip().lower() in ("yes", "y", "true")
    return {"approved": is_approved}


def execute_delete_node(state: DataOpsState) -> dict:
    """Step 3A: Run the actual DELETE inside a safe PostgreSQL transaction."""
    print("\n[Step 3] ✅ Approved! Executing DELETE in PostgreSQL...")
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            query = sql.SQL(
                "DELETE FROM public.{} WHERE status = %s"
            ).format(sql.Identifier(state["table_name"]))
            cur.execute(query, (state["status_filter"],))
            deleted_count = cur.rowcount
        conn.commit()

    return {
        "final_message": (
            f"Successfully deleted {deleted_count} '{state['status_filter']}'"
            f" rows from '{state['table_name']}'."
        )
    }


def cancel_node(state: DataOpsState) -> dict:
    """Step 3B: Safely abort without touching the database."""
    print("\n[Step 3] 🛑 Operation cancelled. No rows were modified.")
    return {
        "final_message": (
            "Cleanup aborted. 0 rows were deleted from"
            f" '{state['table_name']}'."
        )
    }


# --- 4. Define Routing Logic (Conditional Edge) ---
def route_after_approval(state: DataOpsState) -> str:
    if state.get("approved"):
        return "execute_delete"
    return "cancel"


# --- 5. Assemble & Compile the State Graph ---
builder = StateGraph(DataOpsState)

builder.add_node("inspect", parse_and_inspect_node)
builder.add_node("human_approval", human_approval_node)
builder.add_node("execute_delete", execute_delete_node)
builder.add_node("cancel", cancel_node)

builder.add_edge(START, "inspect")
builder.add_edge("inspect", "human_approval")
builder.add_conditional_edges(
    "human_approval",
    route_after_approval,
    {
        "execute_delete": "execute_delete",
        "cancel": "cancel",
    },
)
builder.add_edge("execute_delete", END)
builder.add_edge("cancel", END)

# MemorySaver checkpoints the state so the graph can pause and resume later
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# --- 6. Run the Workflow ---
def main():
    # A thread_id tracks this specific workflow execution in memory
    config = {"configurable": {"thread_id": "cleanup-job-001"}}

    # Start the workflow
    initial_input = {
        "user_request": "Clean up all FAILED rows from the orders table."
    }
    result = graph.invoke(initial_input, config=config)

    # Check if the graph paused at interrupt()
    if "__interrupt__" in result:
        interrupt_info = result["__interrupt__"][0].value
        print(f"\n⚠️  APPROVAL REQUIRED: {interrupt_info['question']}")
        user_reply = input("Enter yes or no: ")

        # Resume the exact paused graph using Command(resume=...)
        final_result = graph.invoke(Command(resume=user_reply), config=config)
        print(f"\nFinal Result: {final_result['final_message']}")
    else:
        print(f"\nFinal Result: {result['final_message']}")


if __name__ == "__main__":
    main()

