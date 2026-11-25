# AgentPOC_04DeepAgent_DynamoDB.py

# [DYNAMODB CHANGE 1]: Added datetime, boto3, and deepagents backend imports
from datetime import datetime, timezone
import os
from typing import TypedDict

import boto3
from boto3.dynamodb.conditions import Key
from deepagents import CompiledSubAgent, create_deep_agent
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    WriteResult,
)
from deepagents.backends.utils import (
    create_file_data,
    file_data_to_string,
    perform_string_replacement,
)
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

# [DYNAMODB CHANGE 2]: Load AWS region and set your DynamoDB table & incident namespace
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
DYNAMO_TABLE_NAME = "deepagent_files"
INCIDENT_NAMESPACE = "orders-incident-2026-11-21"

# Initialize Gemini 3.5 Flash Lite
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)


# =====================================================================
# [DYNAMODB CHANGE 3]: NEW SECTION - DYNAMODB STORAGE BACKEND
# Replaces default in-memory StateBackend so virtual files persist forever
# =====================================================================
def ensure_dynamodb_table(table_name: str, region_name: str) -> None:
    """Creates the DynamoDB table automatically if it doesn't exist yet."""
    client = boto3.client("dynamodb", region_name=region_name)
    if table_name in client.list_tables().get("TableNames", []):
        return

    print(f"Creating DynamoDB table '{table_name}' in {region_name}...")
    client.create_table(
        TableName=table_name,
        KeySchema=[
            {"AttributeName": "namespace", "KeyType": "HASH"},  # Partition Key
            {"AttributeName": "file_path", "KeyType": "RANGE"},  # Sort Key
        ],
        AttributeDefinitions=[
            {"AttributeName": "namespace", "AttributeType": "S"},
            {"AttributeName": "file_path", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    client.get_waiter("table_exists").wait(TableName=table_name)
    print(f"Table '{table_name}' is ready!\n")


class DynamoDBFileBackend(BackendProtocol):
    """Routes Deep Agent's write_file, read_file, edit_file, and ls tools to DynamoDB."""

    def __init__(self, table_name: str, namespace: str, region_name: str = "us-east-1"):
        ensure_dynamodb_table(table_name, region_name)
        self.table = boto3.resource("dynamodb", region_name=region_name).Table(table_name)
        self.namespace = namespace

    def write(self, file_path: str, content: str) -> WriteResult:
        # DeepAgents expects write_file to fail if the file already exists
        existing = self.table.get_item(
            Key={"namespace": self.namespace, "file_path": file_path}
        )
        if "Item" in existing:
            return WriteResult(
                error=f"File '{file_path}' already exists. Use edit_file instead."
            )

        file_data = create_file_data(content)
        self.table.put_item(
            Item={
                "namespace": self.namespace,
                "file_path": file_path,
                "content": file_data["content"],
                "created_at": file_data["created_at"],
                "modified_at": file_data["modified_at"],
            }
        )
        print(f"  💾 [DynamoDB] Saved '{file_path}'")
        return WriteResult(path=file_path, files_update=None)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        res = self.table.get_item(
            Key={"namespace": self.namespace, "file_path": file_path}
        )
        item = res.get("Item")
        if not item:
            return f"Error: File '{file_path}' not found."

        lines = item["content"][offset : offset + limit]
        return "\n".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(lines))

    def edit(
        self, file_path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> EditResult:
        res = self.table.get_item(
            Key={"namespace": self.namespace, "file_path": file_path}
        )
        item = res.get("Item")
        if not item:
            return EditResult(error=f"File '{file_path}' not found.")

        raw_text = file_data_to_string(item)
        updated_text, count = perform_string_replacement(
            raw_text, old_string, new_string, replace_all
        )
        if isinstance(updated_text, str) and updated_text.startswith("Error:"):
            return EditResult(error=updated_text)

        self.table.update_item(
            Key={"namespace": self.namespace, "file_path": file_path},
            UpdateExpression="SET content = :c, modified_at = :m",
            ExpressionAttributeValues={
                ":c": updated_text.splitlines(),
                ":m": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"  ✏️  [DynamoDB] Updated '{file_path}'")
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def ls_info(self, path: str = "/") -> list[FileInfo]:
        res = self.table.query(
            KeyConditionExpression=Key("namespace").eq(self.namespace)
        )
        return [
            {
                "path": item["file_path"],
                "is_dir": False,
                "modified_at": item.get("modified_at", ""),
            }
            for item in res.get("Items", [])
            if item["file_path"].startswith(path)
        ]

    # Required BackendProtocol methods
    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        return []

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.ls_info(path)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return []

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return []


# =====================================================================
# 1. REUSE OUR STAGE 3B CUSTOM WORKFLOW GRAPH AS A COMPILED SUBAGENT
#    (UNCHANGED from your original code)
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

postgres_subagent = CompiledSubAgent(
    name="postgres-auditor",
    description=(
        "Audits PostgreSQL table schemas, row counts, and financial/status"
        " breakdowns."
    ),
    runnable=db_builder.compile(),
)


# =====================================================================
# 2. DEFINE DOMAIN TOOLS FOR THE DEEP AGENT (Infra & FastAPI Logs)
#    (UNCHANGED from your original code)
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


# =====================================================================
# 3. CREATE THE DEEP AGENT (Planning + Virtual Filesystem + Subagents)
# =====================================================================
dynamodb_backend = DynamoDBFileBackend(
table_name=DYNAMO_TABLE_NAME,
namespace=INCIDENT_NAMESPACE,
region_name=AWS_REGION,
)

deep_agent = create_deep_agent(
    model=llm,
    tools=[fetch_infra_metrics, fetch_fastapi_logs],
    subagents=[postgres_subagent],
    # [DYNAMODB CHANGE 5]: Pass backend=dynamodb_backend so files save to AWS instead of RAM
    backend=dynamodb_backend,
    middleware=[TodoListMiddleware()],
    system_prompt=(
    "You are a Principal Site Reliability Engineer conducting a deep"
    " post-incident audit.\nFollow this disciplined workflow:\n1. First,"
    " use write_todos to create a step-by-step investigation plan.\n2."
    " Delegate the PostgreSQL audit to the postgres-auditor subagent, and"
    " gather infra metrics and FastAPI logs using your tools.\n3. Save raw"
    " evidence into /evidence/db_audit.txt and /evidence/telemetry.txt"
    " using write_file so your context stays clean.\n4. Synthesize all"
    " findings and write a polished Markdown runbook to"
    " /incident_report.md using write_file.\n5. Mark all todos"
    " completed and return a brief executive summary."
    ),
)

# =====================================================================
# 4. RUN THE DEEP AGENT & INSPECT FILES STORED IN DYNAMODB
# =====================================================================
def main():
    task = (
    "Run a complete post-incident investigation on the 'orders' table"
    " failures. Audit the database, check infra and FastAPI logs, save"
    " evidence files, and generate /incident_report.md."
    )
    print("🚀 Starting Deep Agent Investigation...\n")

    result = deep_agent.invoke({"messages": [HumanMessage(content=task)]})
    
    # 1. Print the Executive Summary from the final message (UNCHANGED)
    print("=== EXECUTIVE SUMMARY ===")
    print(result["messages"][-1].text)
    
    # [DYNAMODB CHANGE 6]: Replaced result.get("files", {}) with DynamoDB queries!
    # Because files now live in DynamoDB (not in-memory state), we query DynamoDB directly:
    saved_files = dynamodb_backend.ls_info("/")
    print(f"\n=== PERMANENT FILES IN DYNAMODB ({len(saved_files)} files) ===")

    for file_info in saved_files:
        print(f" ☁️  {file_info['path']} (modified: {file_info['modified_at']})")
    
    if any(f["path"] == "/incident_report.md" for f in saved_files):
        print("\n=== CONTENTS OF /incident_report.md (READ FROM DYNAMODB) ===")
        print(dynamodb_backend.read("/incident_report.md"))

if __name__ == "__main__":
    main()