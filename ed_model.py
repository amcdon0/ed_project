#!/usr/bin/env python3
"""
ED Waiting Room Deterioration Model - fully local, trainable (ENG40011, Group 6-2)

A small classifier that reads short sequences of still photos (e.g. one every
5 s) and is trained on your own labelled clips. No cloud
calls: training and inference both run on your machine.

Architecture
    MobileNetV3-Small (ImageNet-pretrained)  -> per-frame features
    GRU over T frames                        -> temporal context
    Linear head, sigmoid output(s)           -> one per state in STATES

Label (column in labels.csv)
    fall_or_collapse

Workflow
    1. record   collect labelled clips from volunteers (webcam)
    2. train    fine-tune on clips, validated on held-out SUBJECTS
    3. eval     per-label sensitivity / specificity / AUROC on a test split
    4. infer    live webcam inference with false-alarm suppression

Install:  pip install torch torchvision pillow numpy opencv-python

Dataset layout
    data/
      labels.csv            clip_id,subject,fall_or_collapse
      clips/<clip_id>/000.jpg 001.jpg ...   (one photo per clip while T_FRAMES = 1)

Include plenty of NEGATIVE clips: upright, resting, sleeping, phone use, bending
down, stretching, walking past. The model's false-alarm rate depends on them.

NOT FOR CLINICAL USE. Research prototype only; collect data with ethics approval
and informed consent.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

STATES = [
    "fall_or_collapse",      # person on the floor / out of the chair / abnormal position
]
T_FRAMES = 1               # photos per prediction. "On the floor" is visible in ONE photo;
                           # persistence over several 5 s photos is handled by ALERT_RULES.
IMG_SIZE = 224

# Alert persistence rules (consecutive positive windows needed, cooldown seconds).
ALERT_RULES = {
    # 2 consecutive positive photos (~5-10 s apart) before alerting, to cut false alarms.
    "fall_or_collapse": {"consecutive": 2, "cooldown_s": 60, "severity": "URGENT"},
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_labels(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["y"] = np.array([float(r[s]) for s in STATES], dtype=np.float32)
    return rows


def split_by_subject(rows: List[dict], val_frac: float, seed: int
                     ) -> Tuple[List[dict], List[dict]]:
    """Split so no person appears in both sets (prevents leakage)."""
    subjects = sorted({r["subject"] for r in rows})
    if len(subjects) < 2:
        sys.exit("Need at least 2 distinct subjects to make a subject-wise split.")
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_frac)))
    val_subj = set(subjects[:n_val])
    train = [r for r in rows if r["subject"] not in val_subj]
    val = [r for r in rows if r["subject"] in val_subj]
    return train, val


class ClipDataset(Dataset):
    def __init__(self, root: Path, rows: List[dict], train: bool):
        self.root, self.rows, self.train = root, rows, train
        norm = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            # Same random params are applied to all T frames of a clip.
            # No hue/saturation jitter, to avoid colour shortcuts.
            self.tf = v2.Compose([
                v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                v2.ToDtype(torch.float32, scale=True),
                norm,
            ])
        else:
            self.tf = v2.Compose([
                v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                norm,
            ])

    def __len__(self):
        return len(self.rows)

    def _pick(self, n: int) -> List[int]:
        if T_FRAMES == 1:
            return [random.randrange(n)] if self.train else [n // 2]
        if n < T_FRAMES:
            raise ValueError(f"clip has {n} frames, need >= {T_FRAMES}")
        idx = np.linspace(0, n - 1, T_FRAMES).round().astype(int)
        return idx.tolist()

    def __getitem__(self, i):
        r = self.rows[i]
        files = sorted((self.root / "clips" / r["clip_id"]).glob("*.jpg"))
        frames = [F.pil_to_tensor(Image.open(files[j]).convert("RGB"))
                  for j in self._pick(len(files))]
        # Resize first so frames can be stacked even if sizes differ.
        frames = [F.resize(f, [IMG_SIZE + 32, IMG_SIZE + 32], antialias=True)
                  for f in frames]
        x = self.tf(torch.stack(frames))  # [T, 3, H, W]
        return x, torch.from_numpy(r["y"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeteriorationNet(nn.Module):
    def __init__(self, pretrained: bool = True, hidden: int = 128):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)
        self.backbone = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru = nn.GRU(576, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, len(STATES))

    def forward(self, x):  # x: [B, T, 3, H, W]
        b, t = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).view(b, t, -1)
        out, _ = self.gru(feats)
        return self.head(self.drop(out[:, -1]))  # logits [B, 5]


def set_backbone_trainable(model: DeteriorationNet, flag: bool):
    for p in model.backbone.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def threshold_for_specificity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Lowest threshold whose specificity >= target (maximises sensitivity)."""
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        return 0.5
    k = min(len(neg) - 1, int(np.ceil(target * len(neg))) - 1)
    return float(min(1.0, neg[max(k, 0)] + 1e-6))


def sens_spec(y: np.ndarray, p: np.ndarray, thr: float) -> Tuple[float, float]:
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fn = np.sum(~pred & (y == 1))
    tn = np.sum(~pred & (y == 0)); fp = np.sum(pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return float(sens), float(spec)


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def report(y: np.ndarray, p: np.ndarray, thresholds: Dict[str, float]) -> float:
    print(f"{'state':26s} {'n+':>4s} {'AUROC':>6s} {'thr':>6s} {'sens':>6s} {'spec':>6s}")
    aucs = []
    for i, s in enumerate(STATES):
        a = auroc(y[:, i], p[:, i])
        se, sp = sens_spec(y[:, i], p[:, i], thresholds[s])
        aucs.append(a)
        print(f"{s:26s} {int(y[:, i].sum()):4d} {a:6.3f} {thresholds[s]:6.2f} "
              f"{se:6.2f} {sp:6.2f}")
    valid = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else 0.0


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cmd_train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    root = Path(a.data)
    rows = read_labels(root / "labels.csv")
    train_rows, val_rows = split_by_subject(rows, a.val_frac, a.seed)
    print(f"{len(train_rows)} train clips / {len(val_rows)} val clips "
          f"(split by subject)")

    dev = device_auto()
    tl = DataLoader(ClipDataset(root, train_rows, True), batch_size=a.batch,
                    shuffle=True, num_workers=a.workers, drop_last=len(train_rows) > a.batch)
    vl = DataLoader(ClipDataset(root, val_rows, False), batch_size=a.batch,
                    shuffle=False, num_workers=a.workers)

    model = DeteriorationNet(pretrained=not a.no_pretrained).to(dev)

    # Class imbalance: weight positives by neg/pos per label.
    Y = np.stack([r["y"] for r in train_rows])
    pos = Y.sum(0); neg = len(Y) - pos
    pos_weight = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0),
                              dtype=torch.float32, device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": a.lr * 0.1},
        {"params": list(model.gru.parameters()) + list(model.head.parameters()),
         "lr": a.lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        set_backbone_trainable(model, ep > a.freeze_epochs)
        model.train()
        tot, t0 = 0.0, time.time()
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        sched.step()

        yv, pv = predict(model, vl, dev)
        aucs = [auroc(yv[:, i], pv[:, i]) for i in range(len(STATES))]
        valid = [v for v in aucs if not np.isnan(v)]
        mean_auc = float(np.mean(valid)) if valid else 0.0
        print(f"epoch {ep:3d}/{a.epochs}  loss {tot / len(train_rows):.4f}  "
              f"val mean AUROC {mean_auc:.3f}  ({time.time() - t0:.0f}s)")

        if mean_auc > best:
            best = mean_auc
            thr = {s: threshold_for_specificity(yv[:, i], pv[:, i], a.target_spec)
                   for i, s in enumerate(STATES)}
            torch.save({"model": model.state_dict(), "thresholds": thr,
                        "frame_interval": a.frame_interval, "t_frames": T_FRAMES,
                        "val_auroc": mean_auc}, a.out)

    print(f"\nBest checkpoint saved to {a.out} (val mean AUROC {best:.3f})")
    ck = torch.load(a.out, map_location=dev)
    model.load_state_dict(ck["model"])
    yv, pv = predict(model, vl, dev)
    print(f"\nValidation results (thresholds tuned for specificity >= {a.target_spec}):")
    report(yv, pv, ck["thresholds"])
    print("\nNOTE: thresholds were tuned on this validation set, so these numbers are "
          "optimistic. Report final performance with `eval` on a separate test set.")


def cmd_eval(a):
    dev = device_auto()
    ck = torch.load(a.checkpoint, map_location=dev)
    model = DeteriorationNet(pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    root = Path(a.data)
    rows = read_labels(root / "labels.csv")
    if a.subjects:
        keep = set(a.subjects.split(","))
        rows = [r for r in rows if r["subject"] in keep]
    loader = DataLoader(ClipDataset(root, rows, False), batch_size=a.batch,
                        num_workers=a.workers)
    y, p = predict(model, loader, dev)
    print(f"{len(rows)} clips")
    report(y, p, ck["thresholds"])


# ---------------------------------------------------------------------------
# Record (data collection)
# ---------------------------------------------------------------------------

def cmd_record(a):
    import cv2
    root = Path(a.data)
    (root / "clips").mkdir(parents=True, exist_ok=True)
    labels = root / "labels.csv"
    new = not labels.exists()
    active = set(a.labels.split(",")) if a.labels else set()
    bad = active - set(STATES)
    if bad:
        sys.exit(f"Unknown labels: {bad}. Valid: {STATES}")

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Could not open camera")
    with open(labels, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["clip_id", "subject"] + STATES)
        for n in range(a.clips):
            input(f"[{n + 1}/{a.clips}] Subject '{a.subject}' ready "
                  f"(labels: {sorted(active) or 'NEGATIVE'}). Enter to record...")
            clip_id = f"{a.subject}_{int(time.time())}_{n}"
            d = root / "clips" / clip_id
            d.mkdir()
            for k in range(a.frames):
                ok, frame = cap.read()
                if ok:
                    cv2.imwrite(str(d / f"{k:03d}.jpg"), frame)
                time.sleep(a.interval)
            w.writerow([clip_id, a.subject] + [int(s in active) for s in STATES])
            f.flush()
            print(f"  saved {clip_id}")
    cap.release()


# ---------------------------------------------------------------------------
# Live inference (all on-device, nothing stored)
# ---------------------------------------------------------------------------

def cmd_infer(a):
    import cv2
    dev = device_auto()
    ck = torch.load(a.checkpoint, map_location=dev)
    model = DeteriorationNet(pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    thr = ck["thresholds"]
    interval = a.interval or ck.get("frame_interval", 5.0)

    tf = v2.Compose([v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
                     v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Could not open camera")

    window: deque = deque(maxlen=T_FRAMES)
    streak = {s: 0 for s in STATES}
    last_alert: Dict[str, float] = {}
    print(f"Running on {dev}. Frame every {interval}s. Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                window.append(F.pil_to_tensor(Image.fromarray(rgb)))
            if len(window) == T_FRAMES:
                x = tf(torch.stack(list(window))).unsqueeze(0).to(dev)
                with torch.no_grad():
                    p = torch.sigmoid(model(x))[0].cpu().numpy()
                now = time.time()
                flags = []
                for i, s in enumerate(STATES):
                    positive = p[i] >= thr[s]
                    streak[s] = streak[s] + 1 if positive else 0
                    if positive:
                        flags.append(f"{s}={p[i]:.2f}")
                    rule = ALERT_RULES[s]
                    if (streak[s] >= rule["consecutive"]
                            and now - last_alert.get(s, float("-inf")) >= rule["cooldown_s"]):
                        last_alert[s] = now
                        print(f"  >>> {rule['severity']} ALERT: {s} (p={p[i]:.2f})")
                print(time.strftime("[%H:%M:%S]"), flags or "no flags")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="collect labelled clips from a webcam")
    r.add_argument("--data", default="data")
    r.add_argument("--subject", required=True, help="anonymous subject ID, e.g. S01")
    r.add_argument("--labels", default="", help="comma list; empty = negative clip")
    r.add_argument("--clips", type=int, default=5)
    r.add_argument("--frames", type=int, default=T_FRAMES)
    r.add_argument("--interval", type=float, default=5.0)
    r.add_argument("--camera", type=int, default=0)
    r.set_defaults(fn=cmd_record)

    t = sub.add_parser("train", help="train the model")
    t.add_argument("--data", default="data")
    t.add_argument("--out", default="ed_model.pt")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--freeze-epochs", type=int, default=5,
                   help="train only GRU+head for the first N epochs")
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-frac", type=float, default=0.25)
    t.add_argument("--target-spec", type=float, default=0.95)
    t.add_argument("--frame-interval", type=float, default=5.0,
                   help="seconds between frames in your clips")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-pretrained", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="evaluate a checkpoint")
    e.add_argument("checkpoint")
    e.add_argument("--data", default="data")
    e.add_argument("--subjects", default="", help="comma list of held-out subjects")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--workers", type=int, default=2)
    e.set_defaults(fn=cmd_eval)

    i = sub.add_parser("infer", help="live webcam inference")
    i.add_argument("checkpoint")
    i.add_argument("--camera", type=int, default=0)
    i.add_argument("--interval", type=float, default=0.0)
    i.set_defaults(fn=cmd_infer)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()