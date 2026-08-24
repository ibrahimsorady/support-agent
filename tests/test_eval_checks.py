"""Offline tests for the eval harness's grading logic (no API calls)."""
from evals.run_evals import check_contains, check_not_contains, check_tools


def test_check_tools_exact_match():
    ok, _ = check_tools(["lookup_order"], ["lookup_order"])
    assert ok


def test_check_tools_mismatch():
    ok, detail = check_tools([], ["lookup_order"])
    assert not ok
    assert "lookup_order" in detail


def test_check_contains_plain_keyword():
    ok, _ = check_contains(["48"], "Refunds are available within 48 hours.")
    assert ok


def test_check_contains_any_of():
    ok, _ = check_contains([["esim", "e-sim"]], "Setting up your eSIM is easy.")
    assert ok


def test_check_contains_missing():
    ok, detail = check_contains(["90"], "The Plus plan costs 50 AED.")
    assert not ok
    assert "90" in detail


def test_check_not_contains_clean():
    ok, _ = check_not_contains(["balance"], "Your order has shipped.")
    assert ok


def test_check_not_contains_hit():
    ok, detail = check_not_contains(["balance"], "Your balance is 45 AED.")
    assert not ok
    assert "balance" in detail
