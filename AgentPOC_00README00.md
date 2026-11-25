
#######################################################################################################################################
# POC Scale : #1 
## Agent Type: Standard - create_agent() / chat.send_message()
## Complexity:  Low 
## Why you have here:  Faster to build; best default for simple lookups and single-turn tasks
##                     It's good for 1-5 tools and 1-5 steps task 

# Logic explanation: 
The POC ues standard agent that uses two tools (schema validation and row count). Here we just used select query, so no issues. 
Suppose if I want to ADD or DELETE rows, then it is not best practice to use them direclty without human approval. 
In order to engage human approvals, standard agent does not help, we should go with 'custom workflow agent' which is next level. 

# Why Stage 1 Breaks & Why We Need Stage 2
• In Stage 1 (Standard Agent): The LLM has total freedom. 
  If you give it a delete_rows() tool, it might execute DELETE FROM orders WHERE status = 'FAILED' immediately 
  without checking how many rows will be wiped out or asking a human engineer first.

• In Stage 2 (Custom Workflow Agent): We take away the LLM's freedom to run destructive commands directly. Instead, we build a State Graph where:
	1. Node 1 (inspect_node): The LLM inspects the schema and counts how many rows match the cleanup criteria.
	2. Node 2 (approval_node — Human-in-the-Loop): The graph pauses execution completely using LangGraph's interrupt(), 
       shows you the exact SQL and affected row count (3 rows), and waits for you to type yes or no.
	3. Conditional Edge: Routes to execute_node only if you approved, or cancel_node if you rejected.

#######################################################################################################################################
# POC Scale : #2
## Agent Type: Custom Workflow Agent (LangGraph State Graph)
## Complexity:  Medium 
## Why you have here:  Needed when steps must follow strict rules, retries, or human approval before writing/mutating data.
##                     It's good for fixed steps, and more tools



#######################################################################################################################################
# POC Scale : #3
## Agent Type: Multi-Agent Router / Supervisor
## Complexity:  High 
## Why you have here:  Needed when one agent gets confused by too many tools across different domains (SQL + FastAPI + Cloud).
##                     It's good for fixed steps, and 15+ tools


#######################################################################################################################################
# POC Scale : #4
## Agent Type: Deep Agent (create_deep_agent)
## Complexity:  Very High 
## Why you have here:  Needed when tasks overflow the context window and require planning lists, virtual files, and subagents.
##                     It's good for 10-50+ steps, and Wide & Open ended tools



AI Agent and Agentic AI — Learning progress
Completed   9/15
Partially completed  4
Pending  2

[x] 1. AI Agent — fundamentals and components:   POC 01  Completed
[x] 2. Agent execution lifecycle:  POC 01                Completed
[x] 3. Tool calling and function calling:  All six POCs  Completed
[x] 4. Agent vs. deterministic workflow: POC 01, 02      Completed
[ ] 5. Agentic AI — autonomy, planning and adaptation:  Adaptation needs further stud    Partially completed
[x] 6. Multi-Agent Architecture and design patterns:   POC 03A, 03B     Completed
[x] 7. Agent orchestration and delegation:  POC 03A, 03B, 04    Completed
[x] 8. State, memory and checkpointing:  POC 02, 04, DynamoDB   Completed
[x] 9. Human-in-the-Loop:  POC 02   Completed
[x] 10. Deep Agents and context management:  POC 04, DynamoDB   Completed
[ ] 11. MCP — Model Context Protocol:  Discussed, not implemented in these POCs   Partially completed
[ ] 12. Agentic RAG:  RAG studied separately; integration pending      Partially completed
[ ] 13. Enterprise security, governance and reliability:  Basic controls implemented     Partially completed
[ ] 14. Token consumption and cost optimization:  Agent-specific optimization pending   Pending
[ ] 15. Architectural objectives and interview preparation: Pending  Pending

