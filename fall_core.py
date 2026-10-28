"""
fall_core.py - model inference and alert logic for the fall-detection backend.

Designed to run on a Raspberry Pi: the main path (OnnxPredictor) needs only
numpy, pillow and onnxruntime - no PyTorch. TorchPredictor is a convenience for
running straight from ed_model.pt on a PC.

Images are processed in memory only and are never written to disk.
"""

from __future__ import annotations

import io
import json
import math
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

IMG_SIZE = 224
PRE_SIZE = IMG_SIZE + 32          # training resized to 256x256, then to 224x224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_rgb(data: bytes):
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("could not decode image") from e
    return img.convert("RGB")


def preprocess(data: bytes) -> np.ndarray:
    """JPEG/PNG bytes -> float32 array shaped [1, 1, 3, 224, 224] (batch, time, C, H, W)."""
    from PIL import Image
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    img = _load_rgb(data)
    # Same two-step resize the training pipeline used.
    img = img.resize((PRE_SIZE, PRE_SIZE), bilinear).resize((IMG_SIZE, IMG_SIZE), bilinear)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None, None].astype(np.float32)


def _sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-v))


class OnnxPredictor:
    """Runs ed_model.onnx (made by export_onnx.py). Threshold comes from ed_model.json."""

    def __init__(self, path: str, threshold: Optional[float] = None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.threshold = threshold if threshold is not None else self._sidecar_threshold(path)

    @staticmethod
    def _sidecar_threshold(path: str) -> float:
        side = Path(path).with_suffix(".json")
        if side.exists():
            return float(json.loads(side.read_text())["threshold"])
        return 0.5

    def predict(self, data: bytes) -> float:
        logit = self.sess.run(None, {self.input_name: preprocess(data)})[0]
        return _sigmoid(float(np.asarray(logit).reshape(-1)[0]))


class TorchPredictor:
    """Runs ed_model.pt directly (needs PyTorch + ed_model.py in the same folder)."""

    def __init__(self, path: str, threshold: Optional[float] = None):
        import torch
        from ed_model import STATES, DeteriorationNet, device_auto
        self.torch = torch
        self.dev = device_auto()
        ck = torch.load(path, map_location=self.dev)
        self.model = DeteriorationNet(pretrained=False).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.threshold = (threshold if threshold is not None
                          else float(ck["thresholds"][STATES[0]]))

    def predict(self, data: bytes) -> float:
        x = self.torch.from_numpy(preprocess(data)).to(self.dev)
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.model(x))[0, 0].item())


def load_predictor(path: str, threshold: Optional[float] = None):
    if path.lower().endswith(".onnx"):
        return OnnxPredictor(path, threshold)
    if path.lower().endswith(".pt"):
        return TorchPredictor(path, threshold)
    raise ValueError("model file must end in .onnx or .pt")


class AlertEngine:
    """
    Turns per-photo probabilities into a yes/no.

    alert      - True while `consecutive` photos in a row were positive
    new_alert  - True once per episode (respecting cooldown): use it to notify staff
    A gap longer than max_gap_s between photos resets the count, so 'consecutive'
    really means consecutive.
    """

    def __init__(self, consecutive: int = 2, cooldown_s: float = 60.0,
                 max_gap_s: float = 30.0):
        self.consecutive, self.cooldown_s, self.max_gap_s = consecutive, cooldown_s, max_gap_s
        self._s: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def update(self, camera_id: str, positive: bool,
               now: Optional[float] = None) -> Tuple[int, bool, bool]:
        now = time.time() if now is None else now
        with self._lock:
            st = self._s.setdefault(
                camera_id, {"streak": 0, "last_seen": None, "last_alert": -math.inf})
            if st["last_seen"] is not None and now - st["last_seen"] > self.max_gap_s:
                st["streak"] = 0
            st["last_seen"] = now
            st["streak"] = st["streak"] + 1 if positive else 0
            alert = st["streak"] >= self.consecutive
            new_alert = alert and (now - st["last_alert"] >= self.cooldown_s)
            if new_alert:
                st["last_alert"] = now
            return st["streak"], alert, new_alert

    def reset(self, camera_id: Optional[str] = None):
        with self._lock:
            if camera_id is None:
                self._s.clear()
            else:
                self._s.pop(camera_id, None)
