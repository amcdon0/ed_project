#!/usr/bin/env python3
"""
Build the training dataset for ed_model.py from the CAUCAFall frame folders.

CAUCAFall layout (as downloaded):
    CAUCAFall/Subject.1/Fall sitting/css100001.png (+ .txt label per frame)
    CAUCAFall/Subject.1/Walking/...   etc.

Step 1 - look at what is in the dataset (changes nothing):
    python make_dataset.py inspect --root "path\\to\\CAUCAFall"

Step 2 - build the dataset, holding out some subjects as an unseen TEST set:
    python make_dataset.py build --root "path\\to\\CAUCAFall" --test-subjects 9,10

Labelling rules
  * Any activity folder whose name contains "fall" is a fall recording; all other
    folders (walking, sitting, kneeling, ...) are NEGATIVE - every frame labelled 0.
  * Inside a fall folder, each frame's .txt starts with 0 (before the fall) or
    1 (the fall has happened). Frames with class 1 are positive (--fall-class 1).
    If the .txt files are missing, it falls back to treating the final --tail
    fraction of frames as positive and warns you.
  * Each saved photo becomes one clip folder, labelled with its subject so that
    training/validation splits never mix the same person.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

STATE = "fall_or_collapse"   # must match STATES in ed_model.py
IMG_EXT = {".png", ".jpg", ".jpeg"}


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def find_activity_dirs(root: Path) -> List[Path]:
    dirs = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file()):
            dirs.append(d)
    return dirs


def subject_of(d: Path, root: Path) -> str:
    for parent in [d] + list(d.parents):
        if parent == root.parent:
            break
        if parent.name.lower().startswith("subject"):
            return parent.name
    return d.parent.name


def subject_number(name: str) -> str:
    m = re.search(r"(\d+)", name)
    return m.group(1) if m else name


def frames_of(d: Path) -> List[Path]:
    return sorted((f for f in d.iterdir() if f.suffix.lower() in IMG_EXT), key=natural_key)


def read_class(img: Path) -> Optional[int]:
    txt = img.with_suffix(".txt")
    if not txt.exists():
        return None
    try:
        for line in txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                return int(float(line.split()[0]))
    except (ValueError, OSError):
        return None
    return None


def imread(path: Path):
    # np.fromfile + imdecode works with Windows paths that contain spaces/unicode.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, img, quality=85):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"encode failed for {path}")
    buf.tofile(str(path))


# ---------------------------------------------------------------------------

def cmd_inspect(a):
    root = Path(a.root)
    dirs = find_activity_dirs(root)
    if not dirs:
        sys.exit(f"No image folders found under {root}. Check --root.")
    print(f"{len(dirs)} activity folders found under {root}\n")
    print(f"{'subject':12s} {'activity':28s} {'imgs':>5s} {'txt':>5s}  txt classes (first / last 20%)")
    subjects = set()
    for d in dirs:
        fr = frames_of(d)
        subjects.add(subject_of(d, root))
        cls = [read_class(f) for f in fr]
        n_txt = sum(c is not None for c in cls)
        tail = cls[int(len(cls) * 0.8):]
        print(f"{subject_of(d, root):12s} {d.name[:28]:28s} {len(fr):5d} {n_txt:5d}  "
              f"{dict(Counter(c for c in cls if c is not None))} / "
              f"{dict(Counter(c for c in tail if c is not None))}")
    print(f"\nSubjects: {sorted(subjects, key=lambda s: int(subject_number(s)) if subject_number(s).isdigit() else 0)}")
    first = next((f.with_suffix('.txt') for d in dirs for f in frames_of(d)[:1]
                  if f.with_suffix('.txt').exists()), None)
    if first:
        print(f"\nSample label file {first.name}:\n  {first.read_text(errors='ignore').strip()[:200]}")


def cmd_build(a):
    root = Path(a.root)
    dirs = find_activity_dirs(root)
    if not dirs:
        sys.exit(f"No image folders found under {root}. Check --root.")

    test_ids = {t.strip() for t in a.test_subjects.split(",") if t.strip()}
    out_train, out_test = Path(a.out), Path(a.test_out)
    for o in (out_train, out_test):
        (o / "clips").mkdir(parents=True, exist_ok=True)

    writers, files = {}, {}
    for key, o in (("train", out_train), ("test", out_test)):
        path = o / "labels.csv"
        new = not path.exists()
        files[key] = open(path, "a", newline="", encoding="utf-8")
        writers[key] = csv.writer(files[key])
        if new:
            writers[key].writerow(["clip_id", "subject", STATE])

    totals = {"train": [0, 0], "test": [0, 0]}  # [positives, negatives]
    warnings = []
    print(f"{'subject':12s} {'activity':24s} {'frames':>6s} {'fall+':>6s}  label source")

    for d in dirs:
        subj = subject_of(d, root)
        split = "test" if (subject_number(subj) in test_ids or subj in test_ids) else "train"
        fr = frames_of(d)
        n = len(fr)
        is_fall = "fall" in d.name.lower()

        if not is_fall:
            flags, source = [False] * n, "negative folder"
        else:
            cls = [read_class(f) for f in fr]
            known = [c for c in cls if c is not None]
            tail = [c for c in cls[int(n * 0.8):] if c is not None]
            if a.fall_class is not None and len(known) >= 0.9 * n:
                # Explicit rule: this class id means "the fall has happened".
                flags, source = [c == a.fall_class for c in cls], f"txt (class {a.fall_class} = fall)"
            elif a.fall_class is None and len(known) >= 0.9 * n and len(set(known)) >= 2 and tail:
                fall_cls = Counter(tail).most_common(1)[0][0]
                flags, source = [c == fall_cls for c in cls], f"txt (auto: fall class={fall_cls})"
            else:
                cut = int(n * (1 - a.tail))
                flags = [i >= cut for i in range(n)]
                source = f"FALLBACK last {int(a.tail * 100)}%"
                warnings.append(f"{subj}/{d.name}: no usable per-frame labels, used fallback")
            frac = sum(flags) / n
            if frac > 0.9 or frac < 0.05:
                warnings.append(f"{subj}/{d.name}: {frac:.0%} of frames positive - check by eye")

        print(f"{subj:12s} {d.name[:24]:24s} {n:6d} {sum(flags):6d}  {source}")

        stem = re.sub(r"[^A-Za-z0-9]+", "", d.name)
        for i in range(0, n, a.every):
            img = imread(fr[i])
            if img is None:
                continue
            h, w = img.shape[:2]
            if w > a.max_width:
                img = cv2.resize(img, (a.max_width, int(h * a.max_width / w)))
            clip_id = f"{subject_number(subj)}_{stem}_{i:05d}"
            cdir = (out_train if split == "train" else out_test) / "clips" / clip_id
            cdir.mkdir(parents=True, exist_ok=True)
            imwrite(cdir / "000.jpg", img)
            writers[split].writerow([clip_id, subj, int(flags[i])])
            totals[split][0 if flags[i] else 1] += 1

    for f in files.values():
        f.close()

    print("\nDone.")
    for k, o in (("train", out_train), ("test", out_test)):
        p, ng = totals[k]
        print(f"  {k:5s} -> {o}/  positives={p}  negatives={ng}")
    for w in warnings:
        print("  WARNING:", w)
    if not test_ids:
        print("  NOTE: no --test-subjects given, so data_test is empty.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("inspect", help="list folders, frame counts and label classes")
    i.add_argument("--root", required=True)
    i.set_defaults(fn=cmd_inspect)

    b = sub.add_parser("build", help="write data/ and data_test/")
    b.add_argument("--root", required=True)
    b.add_argument("--out", default="data")
    b.add_argument("--test-out", default="data_test")
    b.add_argument("--test-subjects", default="",
                   help="comma list of subject numbers kept out of training, e.g. 9,10")
    b.add_argument("--fall-class", type=int, default=1,
                   help="first number in the .txt that means 'fall has happened' (default 1; "
                        "0 = before the fall). Use -1 to auto-detect.")
    b.add_argument("--every", type=int, default=8,
                   help="keep every Nth frame (default 8, about 3 photos/s at 23 fps)")
    b.add_argument("--tail", type=float, default=0.4,
                   help="fallback: fraction of final frames treated as 'on the floor'")
    b.add_argument("--max-width", type=int, default=640)
    b.set_defaults(fn=cmd_build)

    args = ap.parse_args()
    if getattr(args, "fall_class", None) == -1:
        args.fall_class = None
    args.fn(args)


if __name__ == "__main__":
    main()
