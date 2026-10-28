#!/usr/bin/env python3
"""
Score the ONNX model on your held-out test photos, using the same preprocessing
the Pi backend uses. The numbers should be close to `ed_model.py eval`
(AUROC ~0.956, sensitivity ~0.95, specificity ~0.87 earlier). If they are far
off, something changed in the conversion or preprocessing.

    py check_onnx.py ed_model.onnx --data data_test
Works on a PC or the Pi (needs onnxruntime, numpy, pillow only).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from fall_core import OnnxPredictor


def auroc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--data", default="data_test")
    a = ap.parse_args()

    pred = OnnxPredictor(a.model)
    root = Path(a.data)
    rows = list(csv.DictReader(open(root / "labels.csv", newline="", encoding="utf-8")))
    ys, ps = [], []
    for i, r in enumerate(rows):
        files = sorted((root / "clips" / r["clip_id"]).glob("*.jpg"))
        ps.append(pred.predict(files[len(files) // 2].read_bytes()))
        ys.append(int(float(r["fall_or_collapse"])))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    y, p = np.array(ys), np.array(ps)
    pos = p >= pred.threshold
    sens = (pos & (y == 1)).sum() / max((y == 1).sum(), 1)
    spec = (~pos & (y == 0)).sum() / max((y == 0).sum(), 1)
    print(f"\n{len(y)} photos | threshold {pred.threshold:.3f}")
    print(f"AUROC {auroc(y, p):.3f} | sensitivity {sens:.2f} | specificity {spec:.2f}")


if __name__ == "__main__":
    main()
