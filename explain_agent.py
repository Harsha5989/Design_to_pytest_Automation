# explain_agent.py
from typing import Dict, Any
from llm import stream_ollama, run_ollama
from config import EXPLAIN_MODEL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def explain_node(state: Dict[str, Any]):
    messages = state.get("messages", []) or []

    coder_text = ""
    for m in reversed(messages):
        if getattr(m, "name", None) == "coder":
            coder_text = getattr(m, "content", "") or ""
            break
        if isinstance(m, dict) and m.get("name") == "coder":
            coder_text = m.get("content", "") or ""
            break

    # THINK: create an internal analysis/explain plan
    think_prompt = [
        {"role": "system", "content": "You are a technical writer. Produce an internal explanation plan describing what you will explain and sections to include (assumptions, how to run, edge cases)."},
        {"role": "user", "content": f"Code:\n{coder_text}"}
    ]
    try:
        thinking_raw = run_ollama(think_prompt, EXPLAIN_MODEL)
        thinking = str(thinking_raw)
    except Exception as e:
        logger.exception("Explain thinking failed: %s", e)
        thinking = "[Explain thinking failed]"

    # GEN (streamed): generate final explanation
    gen_prompt = [
        {"role": "system", "content": "You are a technical writer. Now produce the full explanation, including how to run the code and assumptions."},
        {"role": "user", "content": f"Plan:\n{thinking}\n\nNow produce the explanation."}
    ]

    acc = ""
    for chunk in stream_ollama(gen_prompt, EXPLAIN_MODEL):
        acc += chunk

    return {
        "messages": [
            {"role": "assistant", "name": "explain_think", "content": str(thinking)},
            {"role": "assistant", "name": "explain", "content": str(acc)}
        ]
    }
