# graph.py
import os
import uuid
import json
import logging
from pathlib import Path
from typing import TypedDict, Annotated, Dict, Any, List, Generator, Tuple

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, add_messages
from langsmith import traceable  # decorator; may raise if langsmith not installed
# Note: we use decorator-level LangSmith tracing only (not LangGraph tracer)

from vision_agent import vision_node
from coder_agent import coder_node
from explain_agent import explain_node
from memory import ConversationMemory

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class JarvisState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    user_image_b64: str | None
    user_audio_bytes: bytes | None
    metadata: Dict[str, Any] | None


def _write_session(session_id: str, partial: Dict[str, Any]):
    fp = OUTPUT_DIR / f"session_{session_id}.json"
    obj = {}
    if fp.exists():
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            obj = {}
    obj.update(partial)
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_trace(fn, name: str):
    """
    Apply langsmith.traceable decorator if available.
    """
    try:
        return traceable(name=name, run_type="llm")(fn)
    except Exception:
        return fn


def build_jarvis_graph():
    # reload .env (app.py may have updated it)
    load_dotenv(".env")
    graph = StateGraph(JarvisState)

    # Add nodes - here we use wrappers that accept state and return dict {"messages":[...]}
    # We decorate with traceable if available (node-level tracing)
    graph.add_node("vision", maybe_trace(vision_node, "vision_node"))
    graph.add_node("coder", maybe_trace(coder_node, "coder_node"))
    graph.add_node("explain", maybe_trace(explain_node, "explain_node"))

    graph.set_entry_point("vision")
    graph.add_edge("vision", "coder")
    graph.add_edge("coder", "explain")
    graph.add_edge("explain", END)

    app = graph.compile()

        # inside graph.build_jarvis_graph()
    def invoke_stream(initial_state: JarvisState) -> Generator[Tuple[str, str, str], None, None]:
        session_id = uuid.uuid4().hex
        memory = ConversationMemory()

        try:
            final_state = app.invoke(initial_state)
        except Exception as e:
            logger.exception("Graph execution failed: %s", e)
            yield ("error", str(e), session_id)
            return

        for m in final_state.get("messages", []):
            node_name = getattr(m, "name", None) or (m.get("name") if isinstance(m, dict) else None)
            content = getattr(m, "content", "") or (m.get("content") if isinstance(m, dict) else "")

            if node_name is None:
                continue

            # If it's a 'thinking' message, emit it as a single block phase: e.g. "vision_think"
            if node_name.endswith("_think"):
                # store and yield whole thinking body once
                memory.add(node_name, content)
                _write_session(session_id, {f"{node_name}_partial": memory.recent(10)})
                yield (node_name, content, session_id)
                continue

            # Otherwise it's a final node output; chunk and stream
            if node_name in ("vision", "coder", "explain"):
                chunk_size = 1024 if node_name == "coder" else 512
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    memory.add(node_name, chunk)
                    _write_session(session_id, {f"{node_name}_partial": memory.recent(10)})
                    yield (node_name, chunk, session_id)

        _write_session(session_id, {"status": "done"})
        yield ("done", "completed", session_id)


    app.invoke_stream = invoke_stream
    return app
