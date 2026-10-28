#!/usr/bin/env python3
"""
Convert the trained PyTorch model (ed_model.pt) to ONNX so it can run on the
Raspberry Pi without PyTorch.   Run this on your PC (where PyTorch is installed).

    py export_onnx.py ed_model.pt
    -> ed_model.onnx   the model
    -> ed_model.json   the alert threshold chosen during training

Then copy ed_model.onnx and ed_model.json to the Pi (same folder as server.py).
Afterwards run check_onnx.py to confirm the exported model scores like the original.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ed_model import STATES, DeteriorationNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="ed_model.pt")
    ap.add_argument("--out", default=None, help="output .onnx path (default: next to checkpoint)")
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(a.checkpoint).with_suffix(".onnx")
    ck = torch.load(a.checkpoint, map_location="cpu")
    model = DeteriorationNet(pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()

    dummy = torch.zeros(1, 1, 3, 224, 224)           # [batch, time=1, C, H, W]
    kwargs = dict(input_names=["image"], output_names=["logit"], opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)   # newer PyTorch
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kwargs)                 # older PyTorch
    print(f"Wrote {out}")

    thr = float(ck["thresholds"][STATES[0]])
    out.with_suffix(".json").write_text(json.dumps({"threshold": thr, "state": STATES[0]}))
    print(f"Wrote {out.with_suffix('.json')} (threshold {thr:.3f})")

    # Self-check: ONNX output must match PyTorch on random input.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed here; skipping the self-check "
              "(py -m pip install onnxruntime)")
        return
    x = torch.randn(1, 1, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print(f"Max difference vs PyTorch: {diff:.2e}  ({'OK' if diff < 1e-3 else 'TOO LARGE - tell me'})")


if __name__ == "__main__":
    main()
