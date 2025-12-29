# audio_agent.py
import tempfile
from transcription import transcribe_local

def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tf.write(audio_bytes)
    tf.flush()
    tf.close()
    text = transcribe_local(tf.name)
    return text
