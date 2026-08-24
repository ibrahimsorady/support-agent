"""Guardrails: the safety layer around the agent.

The system prompt *asks* the model to behave; guardrails *enforce* it. Two
places, two flavors:

  INPUT guardrails  run before the model  -> reject bad requests early
  OUTPUT guardrails run after the model   -> catch unsafe / ungrounded replies

  Deterministic (regex/keywords): cheap, fast, reliable, limited
  Model-based   (a second LLM call): flexible, catches nuance, costs a call

Response strategies used below: BLOCK (input injection), REDACT (card leak),
ESCALATE (ungrounded output).
"""
import re

from openai import OpenAI

from src.config import ENABLE_GROUNDING_GUARD, JUDGE_MODEL

client = OpenAI()

SAFE_INPUT_REFUSAL = (
    "I can only help with account and support questions. "
    "I can't change my instructions or share internal configuration. "
    "How can I help with your account today?"
)
SAFE_OUTPUT_FALLBACK = (
    "I'm not fully certain about that, so I'd rather not guess. "
    "Let me escalate you to a human agent who can confirm the details."
)


# --- INPUT: prompt-injection / instruction-override (deterministic -> BLOCK)
_INJECTION_PATTERNS = [
    r"ignore (all |your |previous |the )*(above |prior )*instructions",
    r"disregard (all |your |previous |the )*(above |prior )*(instructions|rules)",
    r"reveal (your |the )*(system )*(prompt|instructions)",
    r"show me your (system )*(prompt|instructions)",
    r"you are now",
    r"pretend (you are|to be)",
    r"developer mode",
    r"jailbreak",
    r"act as (an? )*(dan|unrestricted)",
]


def check_input(query):
    """Return (allowed, reason, safe_message)."""
    low = query.lower()
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, low):
            return False, "prompt_injection", SAFE_INPUT_REFUSAL
    return True, "", ""


# --- OUTPUT: card-number leak (deterministic -> REDACT) -------------------
# Heuristic: 13-16 digits, optionally split by spaces/dashes. Real systems add
# a Luhn check; this is enough to demonstrate the guardrail. It deliberately
# does NOT match phone numbers (12 digits) or tracking/ticket ids.
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,16}\b")


def _redact_cards(reply):
    if _CARD_RE.search(reply):
        return _CARD_RE.sub("[redacted]", reply), True
    return reply, False


# --- OUTPUT: grounding check (model-based -> ESCALATE, optional) ----------
def _is_grounded(reply, evidence):
    prompt = (
        "You verify whether a support answer is supported by the evidence.\n"
        f"Evidence (knowledge base + tool results):\n{evidence}\n\n"
        f"Answer to check:\n{reply}\n\n"
        "Is every factual claim in the answer supported by the evidence? "
        "Reply YES or NO as the first word."
    )
    resp = client.responses.create(
        model=JUDGE_MODEL,
        instructions="You are a strict grounding checker. Judge only against the evidence.",
        input=prompt,
    )
    return resp.output_text.strip().upper().startswith("YES")


def check_output(reply, context, tool_outputs):
    """Return (possibly_modified_reply, fired_labels)."""
    fired = []
    reply, redacted = _redact_cards(reply)
    if redacted:
        fired.append("output:card_redacted")

    if ENABLE_GROUNDING_GUARD:
        evidence = context + "\n" + "\n".join(tool_outputs)
        if not _is_grounded(reply, evidence):
            fired.append("output:ungrounded")
            return SAFE_OUTPUT_FALLBACK, fired

    return reply, fired
