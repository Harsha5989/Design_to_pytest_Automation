# llm.py
import logging
import time
import json
from typing import List, Dict, Generator, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try to import Ollama or other streaming client. If you use another client adapt the code.
try:
    import ollama
except Exception:
    ollama = None

def _extract_text_from_chunk(chunk: Any) -> str:
    """
    Normalize various chunk shapes into a plain string.
    Handles:
      - plain strings
      - dicts like {'message': {'content': '...'}}
      - dicts like {'choices':[{'delta':{'content':'...'}}]}
      - dicts containing 'content' or 'text'
      - other dicts -> json.dumps
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        try:
            return chunk.decode("utf-8", errors="ignore")
        except Exception:
            return str(chunk)
    if isinstance(chunk, dict):
        # Ollama-style: {"message": {"content": "..."}}
        msg = chunk.get("message")
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        # OpenAI-like streaming deltas
        choices = chunk.get("choices")
        if isinstance(choices, list) and len(choices) > 0:
            # try to extract delta->content or text
            first = choices[0]
            delta = first.get("delta") or first
            if isinstance(delta, dict):
                if "content" in delta:
                    return str(delta["content"])
                if "text" in delta:
                    return str(delta["text"])
        # generic content/text
        if "content" in chunk:
            return str(chunk["content"])
        if "text" in chunk:
            return str(chunk["text"])
        # fallback: dump to JSON
        try:
            return json.dumps(chunk, ensure_ascii=False)
        except Exception:
            return str(chunk)
    # fallback to str()
    return str(chunk)

def stream_ollama(messages: List[Dict[str, str]], model: str, timeout: int = 300) -> Generator[str, None, None]:
    """
    Stream string chunks from the LLM. Always yields plain strings.
    Adapt to your streaming client if necessary.
    """
    logger.info("Start stream for model=%s", model)
    start = time.time()

    if ollama:
        try:
            stream_iter = ollama.chat(model=model, messages=messages, stream=True)
            for chunk in stream_iter:
                text = _extract_text_from_chunk(chunk)
                # ensure string type
                if not isinstance(text, str):
                    text = str(text)
                yield text
        except Exception as e:
            logger.exception("Streaming client failed, falling back to final sync call: %s", e)
            # fallback to sync call if streaming fails
            try:
                resp = ollama.chat(model=model, messages=messages)
                # try to extract content
                if isinstance(resp, dict):
                    content = _extract_text_from_chunk(resp)
                    yield content
                else:
                    yield str(resp)
            except Exception:
                yield "[LLM ERROR]"
    else:
        # Simulated fallback for offline dev: yield text slowly
        text = f"[SIMULATED STREAM: model={model}] " + "This is a simulated streaming response for local development."
        for token in text.split():
            yield token + " "
            time.sleep(0.01)

    logger.info("Stream finished (%.2fs)", time.time() - start)

def run_ollama(messages: List[Dict[str, str]], model: str, timeout: int = 60) -> str:
    """
    Synchronous call that returns a single string response.
    """
    if ollama:
        try:
            resp = ollama.chat(model=model, messages=messages)
            # attempt to extract meaningful text
            return _extract_text_from_chunk(resp)
        except Exception as e:
            logger.exception("run_ollama failed: %s", e)
            return "[LLM SYNC ERROR]"
    # fallback simulation
    return "[SIMULATED SYNC RESPONSE]"
