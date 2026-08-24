"""Function-calling tools: what the model is allowed to *do*.

Two halves:
  1. TOOL_SCHEMAS  — the descriptions the model reads to decide which tool to
     call and what arguments to pass. This is the "menu". Note the FLAT shape
     required by the Responses API (type/name/description/parameters at the top
     level — no nested "function" field).
  2. run_tool()    — the dispatcher that actually executes a chosen tool against
     the mock CRM and returns a result string to feed back to the model.
"""
import json

from mock_backend import crm

TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up the status and details of a customer's order by its order ID.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. 'ORD-1001'."}
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "check_account_status",
        "description": "Check a customer's account status, plan, and outstanding balance by phone number.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "The customer's phone number, e.g. '+971500000001'."}
            },
            "required": ["phone_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "create_ticket",
        "description": "Create a support ticket for an issue that needs human follow-up.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "phone_number": {"type": "string", "description": "The customer's phone number."},
                "summary": {"type": "string", "description": "A short summary of the issue."},
            },
            "required": ["phone_number", "summary"],
            "additionalProperties": False,
        },
    },
]

# Map tool names to the functions that implement them.
_DISPATCH = {
    "lookup_order": lambda a: crm.lookup_order(a.get("order_id", "")),
    "check_account_status": lambda a: crm.check_account_status(a.get("phone_number", "")),
    "create_ticket": lambda a: crm.create_ticket(a.get("phone_number", ""), a.get("summary", "")),
}


def run_tool(name, arguments_json):
    """Execute a tool and return a string result.

    Errors are returned as DATA (not raised) so the model can read them and
    react — e.g. ask the customer for a valid order ID — instead of the whole
    program crashing.
    """
    try:
        args = json.loads(arguments_json) if isinstance(arguments_json, str) else (arguments_json or {})
    except (TypeError, ValueError):
        return json.dumps({"error": "could not parse tool arguments"})

    fn = _DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool '{name}'"})
    return fn(args)
