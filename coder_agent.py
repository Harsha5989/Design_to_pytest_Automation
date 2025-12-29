# coder_agent.py
from typing import Dict, Any
from llm import stream_ollama, run_ollama
from config import CODER_MODEL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def coder_node(state: Dict[str, Any]):
    messages = state.get("messages", []) or []

    # Find last vision final content
    vision_text = ""
    for m in reversed(messages):
        if getattr(m, "name", None) == "vision":
            vision_text = getattr(m, "content", "") or ""
            break
        if isinstance(m, dict) and m.get("name") == "vision":
            vision_text = m.get("content", "") or ""
            break

    user_text = state.get("metadata", {}).get("prompt", "")

    # THINK (synchronous): ask the model to 'plan' code structure, tests, files — return a single block
    think_prompt = [
        {"role": "system", "content": "You are a senior test automation engineer. Provide a full plan for generating production-ready Selenium + PyTest code in Python using POM. This is your internal thinking — produce file list, folder layout, major functions, and edge-case notes."},
        {"role": "user", "content": f"Vision analysis:\n{vision_text}\n\nUser instructions:\n{user_text}"}
    ]
    try:
        thinking_raw = run_ollama(think_prompt, CODER_MODEL)
        thinking = str(thinking_raw)
    except Exception as e:
        logger.exception("Coder thinking failed: %s", e)
        thinking = "[Coder thinking failed]"

    # GEN (streamed): generate the actual code (this may be long) — stream and accumulate
    gen_prompt = [
        {"role": "system", "content": "You are a senior test automation engineer. Now generate runnable code (conftest, POM classes, tests). Include comments and instructions to run."},
        {"role": "user", "content": f"Plan:\n{thinking}\n\nNow produce the code output."}
    ]

    acc = ""
    for chunk in stream_ollama(gen_prompt, CODER_MODEL):
        acc += chunk

    return {
        "messages": [
            {"role": "assistant", "name": "coder_think", "content": str(thinking)},
            {"role": "assistant", "name": "coder", "content": str(acc)}
        ]
    }
