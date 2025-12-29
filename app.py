# app.py
import streamlit as st
import base64
from pathlib import Path
from dotenv import load_dotenv, set_key

from preprocess import preprocess_image_bytes
from audio_agent import transcribe_audio_bytes
from graph import build_jarvis_graph


# -------------------------------------------------------
# .env SETUP
# -------------------------------------------------------
ENV_PATH = Path(".env")
if not ENV_PATH.exists():
    ENV_PATH.touch()

load_dotenv(ENV_PATH)


# -------------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="AI Design → Code Generator", layout="wide")
st.title("🧠 AI Design → Code Generator (LangGraph + LangSmith)")


# -------------------------------------------------------
# SIDEBAR: LANGSMITH CONFIGURATION
# -------------------------------------------------------
st.sidebar.header("⚙️ LangSmith Settings")

existing_env = ENV_PATH.read_text()
existing_key = ""
if "LANGSMITH_API_KEY=" in existing_env:
    try:
        existing_key = existing_env.split("LANGSMITH_API_KEY=")[1].split("\n")[0].strip()
    except:
        existing_key = ""

input_key = st.sidebar.text_input("LangSmith API Key", value=existing_key, type="password")
enable_tracing = st.sidebar.checkbox("Enable LangSmith Tracing", value=bool(existing_key))

if st.sidebar.button("Save API Key"):
    set_key(".env", "LANGSMITH_API_KEY", input_key)
    st.sidebar.success("Saved to .env (no restart needed!)")

if input_key:
    try:
        from langsmith import Client
        Client(api_key=input_key).list_projects()
        st.sidebar.markdown("### 🟢 Valid API Key")
    except:
        st.sidebar.markdown("### 🔴 Invalid API Key")
else:
    st.sidebar.info("Tracing disabled. Enter a key to enable.")


# -------------------------------------------------------
# MAIN UI LAYOUT
# -------------------------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_img = st.file_uploader("📷 Upload UI Image", type=["jpg", "jpeg", "png"])
    uploaded_audio = st.file_uploader("🎤 Upload Audio (optional)", type=["wav", "mp3"])
    user_prompt = st.text_area("📝 Additional Instructions")
    run_btn = st.button("🚀 Run Pipeline")

with col2:
    st.subheader("📡 Live Output")

    # Thinking placeholders
    ph_vision_think = st.empty()
    ph_coder_think = st.empty()
    ph_explain_think = st.empty()

    # Final output placeholders
    ph_vision = st.empty()
    ph_coder = st.empty()
    ph_explain = st.empty()


# Helper to convert uploaded file to bytes
def read_bytes(f):
    if not f:
        return None
    f.seek(0)
    return f.read()


# -------------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------------
def run_pipeline(img_bytes, audio_bytes, user_prompt):
    ph_vision.info("🔍 Processing design...")

    # Build graph
    graph = build_jarvis_graph()

    # Initial state messages
    initial_messages = []

    # AUDIO
    if audio_bytes:
        try:
            text = transcribe_audio_bytes(audio_bytes)
            initial_messages.append({"role": "user", "content": text})
            st.info("🎤 Audio transcribed.")
        except Exception as e:
            st.error(f"Audio transcription error: {e}")

    # USER TEXT
    if user_prompt:
        initial_messages.append({"role": "user", "content": user_prompt})

    # State
    state = {
        "messages": initial_messages,
        "user_image_b64": None,
        "user_audio_bytes": None,
        "metadata": {"prompt": user_prompt},
    }

    # IMAGE
    if img_bytes:
        processed, h = preprocess_image_bytes(img_bytes)
        b64 = base64.b64encode(processed).decode("utf-8")
        state["user_image_b64"] = b64
        state["metadata"]["image_hash"] = h

    # STREAM GRAPH
    stream = graph.invoke_stream(state)

    coder_acc = ""
    explain_acc = ""

    for phase, chunk, _sid in stream:

        # --------------------------------------------------
        # THINKING PHASES
        # --------------------------------------------------
        if phase == "vision_think":
            ph_vision_think.markdown(f"### 🧠 Vision Thinking\n\n{chunk}")
            continue

        if phase == "coder_think":
            ph_coder_think.markdown(f"### 🧠 Coder Thinking\n\n{chunk}")
            continue

        if phase == "explain_think":
            ph_explain_think.markdown(f"### 🧠 Explain Thinking\n\n{chunk}")
            continue

        # --------------------------------------------------
        # FINAL STREAMING OUTPUT
        # --------------------------------------------------
        if phase == "vision":
            ph_vision.markdown(chunk)

        elif phase == "coder":
            coder_acc += chunk
            ph_coder.code(coder_acc, language="python")

        elif phase == "explain":
            explain_acc += chunk
            ph_explain.markdown(explain_acc)

        # --------------------------------------------------
        # ERROR
        # --------------------------------------------------
        elif phase == "error":
            st.error(chunk)

        # --------------------------------------------------
        # DONE
        # --------------------------------------------------
        elif phase == "done":
            ph_vision.success("✨ Completed")


# -------------------------------------------------------
# BUTTON
# -------------------------------------------------------
if run_btn:
    run_pipeline(
        read_bytes(uploaded_img),
        read_bytes(uploaded_audio),
        user_prompt
    )
