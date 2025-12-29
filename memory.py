# memory.py
from typing import List, Dict, Any
import time, logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ConversationMemory:
    def __init__(self, max_items: int = 12):
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = []
        self.summary: str = ""

    def add(self, role: str, content: str):
        self.items.append({"role": role, "content": content, "ts": time.time()})
        if len(self.items) > self.max_items:
            # naive policy: keep last half and mark summary
            self.summary = "[older messages summarized]"
            self.items = self.items[-(self.max_items // 2):]

    def recent(self, n: int = 5):
        return self.items[-n:]