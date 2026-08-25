"""Minimal Streamlit web UI for the support agent.

A thin, clickable face on the existing RAG + tool-calling engine (src/agent.py).
Each turn is stateless on the agent side - only the transcript is kept in
session_state for display; every call to answer() gets just the current query.

Run from the repo root:  streamlit run streamlit_app.py
"""
import time

import streamlit as st

from src.agent import answer
from src.config import VECTOR_BACKEND
from src.metrics import start_metrics_server

start_metrics_server()

st.set_page_config(page_title="Support Agent", page_icon="\U0001F4AC")
st.title("Support Agent")
st.caption(f"Vector backend: `{VECTOR_BACKEND}`")

if "transcript" not in st.session_state:
    st.session_state.transcript = []  # list of {"role", "content", "meta", "latency_ms"}

for turn in st.session_state.transcript:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant":
            meta = turn["meta"]
            with st.expander("Details"):
                st.write("**Sources:** " + (", ".join(meta["sources"]) or "none"))
                st.write("**Tools used:** " + (", ".join(meta["tools_used"]) or "none"))
                st.write("**Guardrails fired:** " + (", ".join(meta["guardrails"]) or "none"))
                st.write(f"**Latency:** {turn['latency_ms']:.0f} ms")

query = st.chat_input("Ask a question...")
if query:
    st.session_state.transcript.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        try:
            start = time.perf_counter()
            reply, meta = answer(query)
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            reply = f"Sorry, something went wrong handling that request: {exc}"
            meta = {"sources": [], "tools_used": [], "guardrails": []}
            latency_ms = 0.0

        st.write(reply)
        with st.expander("Details"):
            st.write("**Sources:** " + (", ".join(meta["sources"]) or "none"))
            st.write("**Tools used:** " + (", ".join(meta["tools_used"]) or "none"))
            st.write("**Guardrails fired:** " + (", ".join(meta["guardrails"]) or "none"))
            st.write(f"**Latency:** {latency_ms:.0f} ms")

    st.session_state.transcript.append({
        "role": "assistant",
        "content": reply,
        "meta": meta,
        "latency_ms": latency_ms,
    })
