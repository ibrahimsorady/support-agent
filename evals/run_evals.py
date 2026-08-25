"""Eval harness: grade the agent against the golden dataset.

Run from the repo root:
    python -m evals.run_evals                     # failures only
    python -m evals.run_evals --verbose           # every check, all cases
    python -m evals.run_evals --trace             # full per-case trace
    python -m evals.run_evals --only order_not_found --trace   # trace one case

Every run also writes evals/last_run.jsonl — a full record of each case (input,
reply, tools, sources, and every check incl. the judge's exact prompt + raw
reply) so you can inspect afterwards.

Grading philosophy — use the cheapest reliable check that will do the job:
    1. expect_tools      structured check on which tools fired  (exact, free)
    2. must_contain      keyword "must appear" check            (cheap, brittle)
    3. must_not_contain  keyword "must NOT appear"              (catch hallucinations)
    4. judge             LLM-as-judge for fuzzy correctness      (a call; last resort)
A case passes only if EVERY check defined on it passes.
"""
import argparse
import json
from pathlib import Path

import yaml
from openai import OpenAI

from src.agent import answer
from src.config import JUDGE_MODEL

_client: OpenAI | None = None
EVALS_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALS_DIR / "cases.yaml"
RESULTS_PATH = EVALS_DIR / "last_run.jsonl"


# --- individual checks: each returns (passed: bool, detail: str) -----------
def check_tools(expected, used):
    exp, got = set(expected), set(used)
    if exp == got:
        return True, f"tools ok ({sorted(got) or 'none'})"
    return False, f"expected {sorted(exp) or 'none'}, got {sorted(got) or 'none'}"


def check_contains(required, reply):
    low = reply.lower()
    missing = []
    for item in required:
        if isinstance(item, list):  # any-of: at least one option must appear
            if not any(opt.lower() in low for opt in item):
                missing.append("any(" + "/".join(item) + ")")
        elif item.lower() not in low:
            missing.append(item)
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "all keywords present"


def check_not_contains(banned, reply):
    low = reply.lower()
    hit = [b for b in banned if b.lower() in low]
    if hit:
        return False, "should not contain: " + ", ".join(hit)
    return True, "no banned terms"


def build_judge_prompt(question, reply, rubric):
    """The exact prompt handed to the judge model. Factored out so the trace
    can show precisely what the judge saw."""
    return (
        "You are grading a telecom customer-support agent's answer.\n"
        f"Customer question: {question}\n"
        f"Agent answer: {reply}\n"
        f"Rubric: {rubric.strip()}\n"
        "Reply with PASS or FAIL as the very first word, then a one-line reason."
    )


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def llm_judge(question, reply, rubric):
    """Returns (passed, short_detail, extra) where extra holds the full prompt
    and raw reply so you can see exactly how the judge decided."""
    prompt = build_judge_prompt(question, reply, rubric)
    resp = _get_client().responses.create(
        model=JUDGE_MODEL,
        instructions="You are a strict but fair grader. Judge only against the rubric.",
        input=prompt,
    )
    raw = " ".join(resp.output_text.split())
    passed = raw.upper().lstrip(":.").startswith("PASS")
    return passed, raw[:160], {"prompt": prompt, "raw": raw}


# --- grade one case: returns a list of check records -----------------------
def grade_case(case, reply, meta):
    checks = []
    if "expect_tools" in case:
        ok, detail = check_tools(case["expect_tools"], meta["tools_used"])
        checks.append({"label": "tools", "passed": ok, "detail": detail})
    if case.get("must_contain"):
        ok, detail = check_contains(case["must_contain"], reply)
        checks.append({"label": "contains", "passed": ok, "detail": detail})
    if case.get("must_not_contain"):
        ok, detail = check_not_contains(case["must_not_contain"], reply)
        checks.append({"label": "not_contains", "passed": ok, "detail": detail})
    if case.get("judge"):
        ok, detail, extra = llm_judge(case["input"], reply, case["judge"])
        checks.append({"label": "judge", "passed": ok, "detail": detail, "extra": extra})
    return checks


def print_trace(rec):
    """Full dump of one case — the answer to 'how do we trace something?'"""
    print(f"\n{'=' * 8} {rec['id']} ({rec['tag']}) {'=' * 8}")
    print(f"Q: {rec['input']}")
    print(f"A: {rec['reply']}")
    print(f"tools: {rec['tools_used'] or 'none'} | sources: {', '.join(rec['sources'])}")
    if rec.get("guardrails"):
        print(f"guardrails fired: {', '.join(rec['guardrails'])}")
    for c in rec["checks"]:
        print(f"  [{'ok' if c['passed'] else 'XX'}] {c['label']}: {c['detail']}")
        if c["label"] == "judge":  # show exactly what the judge saw and said
            print("     --- judge prompt ---")
            for line in c["extra"]["prompt"].splitlines():
                print(f"     {line}")
            print(f"     --- judge raw reply ---\n     {c['extra']['raw']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show every check, all cases")
    parser.add_argument("--trace", action="store_true", help="full per-case trace incl. judge internals")
    parser.add_argument("--only", default=None, help="run only cases whose id contains this string")
    args = parser.parse_args()

    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if args.only in c["id"]]
        if not cases:
            raise SystemExit(f"No case id contains '{args.only}'.")

    print(f"Running {len(cases)} eval case(s) against the agent...\n")

    records = []
    passed_count = 0
    failures = []
    for case in cases:
        reply, meta = answer(case["input"])
        checks = grade_case(case, reply, meta)
        case_passed = all(c["passed"] for c in checks)
        passed_count += int(case_passed)

        rec = {
            "id": case["id"], "tag": case.get("tag", ""), "input": case["input"],
            "reply": reply, "tools_used": meta["tools_used"], "sources": meta["sources"],
            "guardrails": meta.get("guardrails", []),
            "passed": case_passed, "checks": checks,
        }
        records.append(rec)

        if args.trace:
            print_trace(rec)
        else:
            print(f"[{'PASS' if case_passed else 'FAIL'}] {case['id']:<24} ({rec['tag']})")
            for c in checks:
                if args.verbose or not c["passed"]:
                    print(f"   {'ok' if c['passed'] else 'XX'} {c['label']}: {c['detail']}")

        if not case_passed:
            failures.append(case["id"])

    # Always save a full record of the run for later inspection.
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total = len(cases)
    pct = 100 * passed_count / total if total else 0
    print(f"\nScore: {passed_count}/{total} passed ({pct:.0f}%)")
    if failures:
        print("Failures: " + ", ".join(failures))
    print(f"Full trace written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
