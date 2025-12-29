🧠 AI Design → Code Automation Platform
LangGraph • Streamlit • Ollama • Whisper • WebSockets • LangSmith

A fully local, production-grade AI automation system that converts UI designs (images) and voice/text instructions into production-ready test automation code with real-time streaming, memory, caching, and observability.

Key Features
End-to-End LangGraph Pipeline (Vision → Coder → Explain)
Token-by-token streaming
Offline LLMs via Ollama
Offline Whisper transcription
WebSocket streaming backend with auth
Conversation memory & ChromaDB caching
LangSmith observability
VS Code debug-ready
Docker support
Architecture
Streamlit UI → WebSocket → FastAPI Server → LangGraph → Agents → Ollama

Project Structure
Complete_product/
├── app.py
├── server.py
├── graph.py
├── vision_agent.py
├── coder_agent.py
├── explain_agent.py
├── llm.py
├── audio_agent.py
├── preprocess.py
├── memory.py
├── cache.py
├── config.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .env
Requirements
Python 3.10
Ollama running locally
8–16 GB RAM
Linux / macOS / Windows

Models
qwen3-vl:latest
deepseek-coder-v2:16b
deepseek-r1:7b

Pull models:
ollama pull qwen3-vl:latest
ollama pull deepseek-coder-v2:16b
ollama pull deepseek-r1:7b
Environment Variables (.env)
LANGSMITH_API_KEY=your_langsmith_key
AUTH_TOKEN=your_websocket_token
Run Locally
ollama serve
conda create -n automation_env python=3.10
conda activate automation_env
pip install -r requirements.txt
uvicorn server:app --port 8000
streamlit run app.py
Streaming Behavior
Thinking accumulated internally
Final output streamed token-by-token
Optional UI toggle to show thinking
Debugging
VS Code launch.json provided
Breakpoints across agents, graph, server
Full logging support
Docker
docker build -t ai-automation .
docker run -p 8501:8501 -p 8000:8000 ai-automation
Summary
Enterprise-grade, fully offline AI orchestration system with streaming, security, observability, and extensibility.
