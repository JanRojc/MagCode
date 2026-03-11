from typing import List, Optional
import os
from pathlib import Path

os.environ.setdefault("TMPDIR", str(Path("/tmp").resolve()))

import onnxruntime as ort


def require_providers(preferred: List[str]) -> List[str]:
    available = ort.get_available_providers()
    preferred_available = [p for p in preferred if p in available]
    if not preferred_available:
        raise RuntimeError(
            f"Required providers not available. Preferred={preferred}. Available={available}"
        )
    return preferred_available


def create_session(model_path: str, preferred_providers: Optional[List[str]] = None):
    """
    Create an ONNX Runtime session that fails fast if preferred providers are not available.
    """
    if preferred_providers:
        providers = require_providers(preferred_providers)
    else:
        providers = ort.get_available_providers()
    return ort.InferenceSession(model_path, providers=providers)
