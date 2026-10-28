#!/usr/bin/env python3
"""
Send photos to the running server, one every few seconds, and print the answers.
Use this to test the backend before the real camera is connected.

    py send_test.py --folder data_test/clips --url http://localhost:8000 --interval 1 --limit 20
    py send_test.py --folder some_photos --url http://192.168.1.50:8000

Replay ONE recording in order (the folder names come from make_dataset.py):
    py send_test.py --folder data_test/clips --match Fallsitting --limit 40 --interval 1
    py send_test.py --folder data_test/clips --match Walk --limit 40 --interval 1

Uses only the Python standard library.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="folder of .jpg/.png (searched recursively)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--camera", default="test-cam")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--match", default="", help="only photos whose folder name contains this text")
    ap.add_argument("--skip", type=int, default=0, help="skip this many photos first")
    a = ap.parse_args()

    files = sorted(p for p in Path(a.folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if a.match:
        files = [p for p in files if a.match.lower() in p.parent.name.lower()]
    if not files:
        raise SystemExit(f"No images found in {a.folder} matching '{a.match}'")
    for f in files[a.skip: a.skip + a.limit]:
        req = urllib.request.Request(f"{a.url}/predict?camera_id={a.camera}",
                                     data=f.read_bytes(), headers={"Content-Type": "image/jpeg"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read())
        print(f"{f.parent.name}/{f.name}: p={r['probability']:.2f} "
              f"alert={'YES' if r['alert'] else 'no'} ({r['consecutive']}/{r['needed']})"
              f"{'  <-- NEW ALERT' if r['new_alert'] else ''}")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()