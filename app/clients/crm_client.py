"""HTTP client for the CRM service, with timeouts, retries, and errors-as-data."""
import json
import time

import httpx

from app.config import CRM_API_URL

_client = httpx.Client(base_url=CRM_API_URL, timeout=httpx.Timeout(5.0, connect=2.0))

_RETRIES = 2          # total attempts = 1 + _RETRIES
_BACKOFF = 0.3        # seconds, grows each retry


def _get(path: str) -> str:
    return _request("GET", path)


def _post(path: str, payload: dict) -> str:
    return _request("POST", path, payload)


def _request(method: str, path: str, payload: dict | None = None) -> str:
    """Call the CRM. On failure, return an error STRING (never raise) so the
    agent can respond gracefully instead of the turn crashing."""
    last_err = None
    for attempt in range(_RETRIES + 1):
        try:
            if method == "GET":
                r = _client.get(path)
            else:
                r = _client.post(path, json=payload)
            r.raise_for_status()
            return r.text
        except httpx.HTTPStatusError as e:
            # 4xx = our bad request; don't retry, report it
            if 400 <= e.response.status_code < 500:
                return json.dumps({"error": f"CRM returned {e.response.status_code}"})
            last_err = e                                  # 5xx: worth retrying
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_err = e                                  # network/timeout: retry
        if attempt < _RETRIES:
            time.sleep(_BACKOFF * (attempt + 1))
    return json.dumps({"error": "CRM service unavailable, please try again later"})


def lookup_order(order_id: str) -> str:
    return _get(f"/orders/{order_id}")


def check_account_status(phone_number: str) -> str:
    return _get(f"/customers/{phone_number}")


def create_ticket(phone_number: str, summary: str) -> str:
    return _post("/tickets", {"phone_number": phone_number, "summary": summary})