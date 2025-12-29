# preprocess.py
from PIL import Image, ImageOps
import io, hashlib
from typing import Tuple

def preprocess_image_bytes(img_bytes: bytes, max_size: int = 1024, quality: int = 100) -> Tuple[bytes, str]:
    """
    Resize, convert to RGB JPEG, compress, and return (bytes, sha256).
    """
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_size, max_size))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality, optimize=True)
        data = out.getvalue()
        h = hashlib.sha256(data).hexdigest()
        return data, h
