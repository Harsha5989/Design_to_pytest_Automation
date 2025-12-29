# transcription.py
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from faster_whisper import WhisperModel
    HAS_FAST = True
except Exception:
    HAS_FAST = False

try:
    import whisper
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

from config import WHISPER_MODEL

def transcribe_local(path: str) -> str:
    """
    Transcribe using faster-whisper or whisper. Returns text.
    """
    logger.info("Transcribing: %s", path)
    if HAS_FAST:
        model = WhisperModel(WHISPER_MODEL, device="cpu")
        segments, _ = model.transcribe(path)
        return " ".join([s.text for s in segments])
    elif HAS_WHISPER:
        model = whisper.load_model(WHISPER_MODEL)
        r = model.transcribe(path)
        return r.get("text", "")
    else:
        raise RuntimeError("Install faster-whisper or whisper for local transcription")
