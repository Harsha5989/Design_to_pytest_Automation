# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env
ENV_PATH = Path(".env")
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Output Directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_DIR = Path("db")
DB_DIR.mkdir(parents=True, exist_ok=True)

# Model names (override with environment vars)
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:latest")
CODER_MODEL = os.getenv("CODER_MODEL", "deepseek-coder-v2:16b")
EXPLAIN_MODEL = os.getenv("EXPLAIN_MODEL", "deepseek-r1:7b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large")

# LangSmith (loaded automatically)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
