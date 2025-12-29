# cache.py
import hashlib, json, logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

DB_DIR = Path("db")
CHROMA_DIR = DB_DIR / "chroma"
FS_CACHE_DIR = DB_DIR / "fs_cache"
FS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class SimpleCache:
    def __init__(self):
        self.use_chroma = False
        if chromadb:
            try:
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                self.col = client.get_or_create_collection("responses")
                self.use_chroma = True
                logger.info("Chromadb enabled.")
            except Exception:
                logger.exception("Chromadb init failed; falling back to FS cache.")
                self.use_chroma = False

    def _key(self, image_hash: str, prompt: str) -> str:
        s = (image_hash or "") + "|" + (prompt or "")
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def get(self, image_hash: str, prompt: str):
        key = self._key(image_hash, prompt)
        if self.use_chroma:
            try:
                res = self.col.query(query_texts=[prompt], n_results=1)
                if res and res.get("metadatas"):
                    md = res["metadatas"][0]
                    return md.get("response")
            except Exception:
                logger.exception("Chromadb query failed")
        fp = FS_CACHE_DIR / f"{key}.json"
        if fp.exists():
            try:
                return json.loads(fp.read_text(encoding="utf-8")).get("response")
            except Exception:
                logger.exception("FS cache read failed")
        return None

    def set(self, image_hash: str, prompt: str, response: str):
        key = self._key(image_hash, prompt)
        if self.use_chroma:
            try:
                self.col.add(
                    ids=[key],
                    metadatas=[{"response": {"thinking":[response.thinking] if hasattr(response, 'thinking') else response,
                               "thinking":[response.content] if hasattr(response, 'content') else response
                               }, "prompt": prompt, "image_hash": image_hash}],
                               
                    documents={"thinking":[response.thinking] if hasattr(response, 'thinking') else response,
                               "thinking":[response.content] if hasattr(response, 'content') else response
                               }
                )
                return
            except Exception:
                logger.exception("Chromadb add failed")
        
        # Fallback to File System Cache
        fp = FS_CACHE_DIR / f"{key}.json"
        try:
            fp.write_text(json.dumps({"response": response, "prompt": prompt, "image_hash": image_hash}, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("FS cache write failed")

cache = SimpleCache()
