"""
alerting.py - turns the detector's per-photo yes/no into alerts a nurse can act on.

Responsibilities
  * One ALERT per fall episode (not one per photo): created when the detector
    first says "yes", then tracked as  active -> acknowledged -> resolved.
  * Reminders for alerts nobody has acknowledged yet.
  * Camera health: if a camera stops sending photos, that is itself an alert
    ("camera_offline") - a silent camera must never look like "all clear".
  * Live push to connected phones (via the Hub, served as Server-Sent Events).
  * History saved to a small JSON file so a restart doesn't lose open alerts.

Only flags and times are stored. Photos are never kept.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

RESOLUTIONS = {"assisted", "false_alarm", "other"}


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


# ---------------------------------------------------------------------------
# Live event hub (thread-safe publish -> asyncio subscribers)
# ---------------------------------------------------------------------------

class Hub:
    """Publish events from any thread; subscribers are asyncio queues."""

    def __init__(self):
        self._subs: set = set()
        self._lock = threading.Lock()

    def subscribe(self) -> Tuple[asyncio.AbstractEventLoop, asyncio.Queue]:
        sub = (asyncio.get_running_loop(), asyncio.Queue(maxsize=200))
        with self._lock:
            self._subs.add(sub)
        return sub

    def unsubscribe(self, sub):
        with self._lock:
            self._subs.discard(sub)

    @staticmethod
    def _put(q: asyncio.Queue, item):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass  # a stalled client; it gets a fresh snapshot when it reconnects

    def publish(self, event: str, data: dict):
        with self._lock:
            subs = list(self._subs)
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(self._put, q, (event, data))
            except RuntimeError:
                pass  # that client's event loop has closed

    def count(self) -> int:
        with self._lock:
            return len(self._subs)


# ---------------------------------------------------------------------------
# Alert manager
# ---------------------------------------------------------------------------

@dataclass
class CameraState:
    camera_id: str
    last_seen: Optional[float] = None
    online: bool = True
    open_alert_id: Optional[int] = None      # open fall alert, if any
    offline_alert_id: Optional[int] = None
    clear_count: int = 0                     # consecutive photos with no fall
    suppressed: bool = False                 # nurse resolved; wait for the scene to clear
    last_probability: Optional[float] = None


class AlertManager:
    def __init__(self, hub: Optional[Hub] = None, path: Optional[str] = "alerts.json",
                 offline_after: float = 30.0, remind_every: float = 60.0,
                 clear_after: int = 3, webhook_url: Optional[str] = None,
                 max_history: int = 500):
        self.hub = hub or Hub()
        self.path = path
        self.offline_after = offline_after
        self.remind_every = remind_every
        self.clear_after = clear_after
        self.webhook_url = webhook_url
        self.max_history = max_history

        self._lock = threading.RLock()
        self._alerts: Dict[int, dict] = {}
        self._cams: Dict[str, CameraState] = {}
        self._next_id = 1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._load()

    # ---- persistence ------------------------------------------------------

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for a in data.get("alerts", []):
                self._alerts[a["id"]] = a
                if a["type"] == "fall" and a["status"] != "resolved":
                    self._cam(a["camera_id"]).open_alert_id = a["id"]
            self._next_id = max(data.get("next_id", 1),
                                max(self._alerts, default=0) + 1)
            n_open = sum(1 for a in self._alerts.values() if a["status"] != "resolved")
            print(f"Loaded {len(self._alerts)} alerts from {self.path} ({n_open} still open)",
                  flush=True)
        except (OSError, ValueError, KeyError) as e:
            print(f"WARNING: could not read {self.path} ({e}); starting with no history",
                  flush=True)

    def _save(self):
        if not self.path:
            return
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"next_id": self._next_id,
                           "alerts": list(self._alerts.values())}, f)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"WARNING: could not save alerts ({e})", flush=True)

    # ---- helpers ----------------------------------------------------------

    def _cam(self, camera_id: str) -> CameraState:
        if camera_id not in self._cams:
            self._cams[camera_id] = CameraState(camera_id)
        return self._cams[camera_id]

    @staticmethod
    def _public(a: dict) -> dict:
        return {k: v for k, v in a.items() if not k.startswith("_")}

    def _camera_view(self, cam: CameraState, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        return {"camera_id": cam.camera_id, "online": cam.online,
                "last_seen": _iso(cam.last_seen) if cam.last_seen else None,
                "seconds_since_last_photo":
                    round(now - cam.last_seen, 1) if cam.last_seen else None,
                "last_probability": cam.last_probability,
                "open_alert_id": cam.open_alert_id}

    def _emit(self, event: str, data: dict):
        self.hub.publish(event, data)
        if self.webhook_url and event in ("alert", "reminder"):
            threading.Thread(target=self._post_webhook, args=(event, data),
                             daemon=True).start()

    def _post_webhook(self, event: str, data: dict):
        try:
            req = urllib.request.Request(
                self.webhook_url, data=json.dumps({"event": event, "alert": data}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
        except Exception as e:  # never let a dead webhook affect detection
            print(f"webhook failed: {e}", flush=True)

    def _create(self, type_: str, camera_id: str, probability: Optional[float],
                now: float) -> dict:
        a = {"id": self._next_id, "type": type_, "camera_id": camera_id,
             "status": "active", "created_at": _iso(now),
             "acknowledged_at": None, "acknowledged_by": None,
             "resolved_at": None, "resolved_by": None, "resolution": None,
             "probability": probability, "peak_probability": probability,
             "still_detected": type_ == "fall", "reminders": 0,
             "_last_notified": now}
        self._alerts[a["id"]] = a
        self._next_id += 1
        self._trim()
        self._save()
        self._emit("alert", self._public(a))
        return a

    def _trim(self):
        resolved = sorted((i for i, a in self._alerts.items() if a["status"] == "resolved"))
        for i in resolved[:max(0, len(resolved) - self.max_history)]:
            del self._alerts[i]

    def _close(self, alert_id: int, resolution: str, by: Optional[str], now: float):
        a = self._alerts[alert_id]
        a["status"], a["resolution"] = "resolved", resolution
        a["resolved_at"], a["resolved_by"] = _iso(now), by
        self._save()
        self._emit("alert_update", self._public(a))

    # ---- called for every photo ------------------------------------------

    def register_camera(self, camera_id: str, now: Optional[float] = None):
        """Expect this camera: if it never reports, that raises a camera_offline alert."""
        now = time.time() if now is None else now
        with self._lock:
            cam = self._cam(camera_id)
            if cam.last_seen is None:
                cam.last_seen = now

    def on_photo(self, camera_id: str, probability: float, alert_state: bool,
                 now: Optional[float] = None) -> Optional[int]:
        """alert_state = the detector's yes/no for this photo. Returns open fall alert id."""
        now = time.time() if now is None else now
        with self._lock:
            cam = self._cam(camera_id)
            cam.last_seen, cam.last_probability = now, probability

            if not cam.online:
                cam.online = True
                self._emit("camera", self._camera_view(cam, now))
                if cam.offline_alert_id is not None:
                    self._close(cam.offline_alert_id, "camera_back", None, now)
                    cam.offline_alert_id = None

            a = self._alerts.get(cam.open_alert_id) if cam.open_alert_id else None
            if a is not None and a["status"] == "resolved":
                a, cam.open_alert_id = None, None

            if alert_state:
                cam.clear_count = 0
                if a is None:
                    if not cam.suppressed:
                        cam.open_alert_id = self._create("fall", camera_id, probability, now)["id"]
                else:
                    a["probability"] = probability
                    a["peak_probability"] = max(a["peak_probability"] or 0.0, probability)
                    if not a["still_detected"]:
                        a["still_detected"] = True
                        self._save()
                        self._emit("alert_update", self._public(a))
            else:
                cam.clear_count += 1
                if cam.clear_count >= self.clear_after:
                    cam.suppressed = False
                    if a is not None and a["still_detected"]:
                        a["still_detected"] = False
                        self._save()
                        self._emit("alert_update", self._public(a))
            return cam.open_alert_id

    # ---- nurse actions ----------------------------------------------------

    def acknowledge(self, alert_id: int, by: Optional[str] = None,
                    now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            a = self._alerts.get(alert_id)
            if a is None:
                raise KeyError(alert_id)
            if a["status"] == "resolved":
                raise ValueError("alert is already resolved")
            if a["status"] == "active":
                a["status"] = "acknowledged"
                a["acknowledged_at"], a["acknowledged_by"] = _iso(now), by
                self._save()
                self._emit("alert_update", self._public(a))
            return self._public(a)

    def resolve(self, alert_id: int, reason: str = "assisted", by: Optional[str] = None,
                now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        if reason not in RESOLUTIONS:
            raise ValueError(f"reason must be one of {sorted(RESOLUTIONS)}")
        with self._lock:
            a = self._alerts.get(alert_id)
            if a is None:
                raise KeyError(alert_id)
            if a["status"] == "resolved":
                raise ValueError("alert is already resolved")
            self._close(alert_id, reason, by, now)
            cam = self._cams.get(a["camera_id"])
            if cam is not None:
                if cam.open_alert_id == alert_id:
                    cam.open_alert_id = None
                    # Don't re-alert for the same person while they are still being
                    # detected; wait until the scene has been clear for a few photos.
                    cam.suppressed = cam.clear_count < self.clear_after
                if cam.offline_alert_id == alert_id:
                    cam.offline_alert_id = None
            return self._public(a)

    # ---- periodic checks: offline cameras and reminders -------------------

    def check(self, now: Optional[float] = None):
        now = time.time() if now is None else now
        with self._lock:
            for cam in self._cams.values():
                if (cam.online and cam.last_seen is not None
                        and now - cam.last_seen > self.offline_after):
                    cam.online = False
                    self._emit("camera", self._camera_view(cam, now))
                    cam.offline_alert_id = self._create(
                        "camera_offline", cam.camera_id, None, now)["id"]
            for a in self._alerts.values():
                if a["status"] == "active" and now - a["_last_notified"] >= self.remind_every:
                    a["reminders"] += 1
                    a["_last_notified"] = now
                    self._save()
                    self._emit("reminder", self._public(a))

    def start(self, every: float = 2.0):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop():
            while not self._stop.wait(every):
                try:
                    self.check()
                except Exception as e:
                    print(f"alert monitor error: {e}", flush=True)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ---- read views -------------------------------------------------------

    def get(self, alert_id: int) -> Optional[dict]:
        with self._lock:
            a = self._alerts.get(alert_id)
            return self._public(a) if a else None

    def list_alerts(self, status: Optional[str] = None, limit: int = 50) -> List[dict]:
        with self._lock:
            items = sorted(self._alerts.values(), key=lambda a: a["id"], reverse=True)
            if status == "open":
                items = [a for a in items if a["status"] != "resolved"]
            elif status:
                items = [a for a in items if a["status"] == status]
            return [self._public(a) for a in items[:limit]]

    def cameras(self) -> List[dict]:
        with self._lock:
            now = time.time()
            return [self._camera_view(c, now) for c in self._cams.values()]

    def snapshot(self) -> dict:
        return {"server_time": _iso(time.time()),
                "alerts": self.list_alerts("open", 200),
                "cameras": self.cameras()}