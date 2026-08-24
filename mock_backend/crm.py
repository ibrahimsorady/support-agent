"""A tiny fake CRM standing in for real systems (Salesforce, billing, etc.).

In production these functions would call those systems' APIs. Here they read
from in-memory dictionaries so you can develop and test the agent with no real
backend. Each function returns a JSON string — structured data the model can
parse reliably, rather than free-form prose.
"""
import json
import random

# --- Fake data ------------------------------------------------------------
_ORDERS = {
    "ORD-1001": {"item": "iPhone 15 (128GB)", "status": "shipped",
                 "eta": "2 days", "tracking": "MM-8842301"},
    "ORD-1002": {"item": "eSIM activation", "status": "processing",
                 "eta": "30 minutes", "tracking": None},
    "ORD-1003": {"item": "5G router", "status": "delivered",
                 "eta": "-", "tracking": "MM-8842277"},
}

_ACCOUNTS = {
    "+971500000001": {"name": "Sara", "plan": "Plus", "balance_aed": 0.0, "status": "active"},
    "+971500000002": {"name": "Omar", "plan": "Unlimited", "balance_aed": 45.0, "status": "active"},
    "+971500000003": {"name": "Layla", "plan": "Lite", "balance_aed": 120.0, "status": "suspended"},
}


# --- "API" functions ------------------------------------------------------
def lookup_order(order_id):
    order = _ORDERS.get(order_id.strip().upper())
    if not order:
        return json.dumps({"found": False, "order_id": order_id})
    return json.dumps({"found": True, "order_id": order_id.strip().upper(), **order})


def check_account_status(phone_number):
    acct = _ACCOUNTS.get(phone_number.strip())
    if not acct:
        return json.dumps({"found": False, "phone_number": phone_number})
    return json.dumps({"found": True, "phone_number": phone_number.strip(), **acct})


def create_ticket(phone_number, summary):
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    return json.dumps({"created": True, "ticket_id": ticket_id,
                       "phone_number": phone_number.strip(), "summary": summary})
