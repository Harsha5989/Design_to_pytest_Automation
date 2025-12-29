# vision_agent.py
from typing import Dict, Any, List
from llm import stream_ollama, run_ollama
from cache import cache
from config import VISION_MODEL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def vision_node(state: Dict[str, Any]):
    """
    1) Create a 'thinking' string (full, synchronous) that contains the model's analysis/chain-of-thought.
    2) Then produce the final analysis/content (accumulated string).
    3) Return two messages: a single 'vision_think' message, then the 'vision' final message.
    """
    messages = state.get("messages", []) or []
    # build a single user prompt joined from incoming user messages
    user_text_parts = []
    for m in messages:
        try:
            if getattr(m, "type", None) in ("user", "human") or getattr(m, "name", None) == "user":
                user_text_parts.append(m.content or "")
        except Exception:
            if isinstance(m, dict):
                user_text_parts.append(m.get("content", ""))
    user_prompt = "\n".join([p for p in user_text_parts if p]).strip()
    user_prompt = user_prompt or state.get("metadata", {}).get("prompt", "")

    img_b64 = state.get("user_image_b64")
    image_hash = state.get("metadata", {}).get("image_hash")

    # If full cached final exists, return think + final from cache quickly
    cached = cache.get(image_hash, user_prompt)
    if cached:
        thinking = "[cached analysis — showing full analysis]"
        return {
            "messages": [
                {"role": "assistant", "name": "vision_think", "content": str(thinking)},
                {"role": "assistant", "name": "vision", "content": str(cached)}
            ]
        }

    # THINK (synchronous): ask the model to "think" / analyze fully and return one block
    think_prompt = [
        {"role": "system", "content": "You are a senior UI/UX analyst. Produce a complete internal analysis. Do NOT include final code; this is your private thinking summary."},
        {"role": "user", "content": f"Instructions / Context:\n{user_prompt}\n\nImage present: {'yes' if img_b64 else 'no'}"}
    ]

    # use run_ollama (sync) to get full thinking as one string
    try:
        thinking_raw = run_ollama(think_prompt, VISION_MODEL)
        # run_ollama may return string or dict; ensure string
        thinking = str(thinking_raw)
    except Exception as e:
        logger.exception("Vision thinking sync failed: %s", e)
        thinking = "[Vision thinking failed]"

    # GEN (streamed): Now produce the final analysis/result (we will accumulate to return)
    gen_prompt = [
        {"role": "system", "content": "You are a senior UI/UX analyst. Now produce the final analysis output (short, actionable items, components, labels, structure)."},
        {"role": "user", "content": f"Context:\n{user_prompt}\n\nPlease produce final structured output based on your analysis."}
    ]

    acc = ""
    for chunk in stream_ollama(gen_prompt, VISION_MODEL):
        acc += chunk

    # Save to cache (final output)
    try:
        cache.set(image_hash, user_prompt, acc)
    except Exception:
        logger.exception("Vision cache write failed")

    return {
        "messages": [
            {"role": "assistant", "name": "vision_think", "content": str(thinking)},
            {"role": "assistant", "name": "vision", "content": str(acc)}
        ]
    }
