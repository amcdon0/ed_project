#!/usr/bin/env python3
"""
Fall-detection backend: camera photo in -> alerts to nurses' phones.

Run on the Raspberry Pi:
    ED_API_KEY=choose-a-secret ED_CAPTURE_INTERVAL=5 \
    uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3

PHOTOS IN (either or both)
  A) the Pi captures its own photos:  ED_CAPTURE_INTERVAL=5
  B) another device POSTs a JPEG:      POST /predict?camera_id=cam1

ALERTS OUT  (for the phone frontend)
  GET  /events                    live stream (Server-Sent Events) - the main feed
  GET  /alerts?status=open        list alerts (status: open|active|acknowledged|resolved)
  GET  /alerts/{id}
  POST /alerts/{id}/acknowledge   body (optional): {"by": "Nurse A"}
  POST /alerts/{id}/resolve       body: {"reason": "assisted|false_alarm|other", "by": "Nurse A"}
  GET  /cameras                   online/offline status of each camera
  GET  /health                    no key needed

/events sends these named events (each `data:` line is JSON):
  snapshot      sent on connect: {alerts: [open alerts], cameras: [...], server_time}
  alert         a NEW alert was created        -> buzz / show full-screen alert
  reminder      an alert is still unacknowledged -> buzz again
  alert_update  status changed (acknowledged / resolved / no longer detected)
  camera        a camera went offline or came back
  heartbeat     every 15 s, so the phone can tell the connection is alive

Alert types: "fall" and "camera_offline" (a silent camera is never treated as 'all clear').

Settings (environment variables)
  ED_API_KEY           shared secret. Send as header  X-API-Key  or query  ?key=
                       (EventSource in browsers can't set headers, so use ?key= there).
                       If unset the API is OPEN - fine for a private demo only.
  ED_CORS_ORIGINS      allowed web origins, comma separated (default "*")
  ED_MODEL             model file (default ed_model.onnx)
  ED_THRESHOLD         override the saved detection threshold
  ED_CONSECUTIVE       consecutive positive photos before "yes" (default 2)
  ED_CAPTURE_INTERVAL  seconds between self-captured photos, 0 = off (default 0)
  ED_OFFLINE_AFTER     seconds without a photo before camera_offline (default 30)
  ED_REMIND_EVERY      seconds between reminders for unacknowledged alerts (default 60)
  ED_CLEAR_AFTER       photos with no fall before the scene counts as clear (default 3)
  ED_EXPECTED_CAMERAS  comma list of camera ids that must report (alerts if they never do)
  ED_ALERTS_FILE       where alert history is kept (default alerts.json)
  ED_WEBHOOK_URL       optional: POST each new alert / reminder here as JSON

Photos are processed in memory and never saved. Only flags and times are stored.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from alerting import AlertManager
from fall_core import AlertEngine, load_predictor

MAX_BYTES = 10 * 1024 * 1024
SELF_CAMERA_ID = "pi-camera"
HEARTBEAT_S = 15.0


def env_float(name: str, default: Optional[float]) -> Optional[float]:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# ---------------------------------------------------------------------------
# Camera (only used when ED_CAPTURE_INTERVAL > 0)
# ---------------------------------------------------------------------------

def make_camera() -> Callable[[], bytes]:
    """Return a function that grabs one JPEG (as bytes). Pi camera first, else USB webcam."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        Picamera2 = None

    if Picamera2 is not None:
        cam = Picamera2()
        cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
        cam.start()
        time.sleep(2)  # let exposure settle

        def grab() -> bytes:
            buf = io.BytesIO()
            cam.capture_file(buf, format="jpeg")
            return buf.getvalue()
        return grab

    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("No camera library: install picamera2 (Pi camera) "
                           "or opencv-python (USB webcam).") from e
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0")

    def grab() -> bytes:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()
    return grab


def capture_loop(grab: Callable[[], bytes], process: Callable[[str, bytes], dict],
                 interval: float, stop: threading.Event):
    while not stop.is_set():
        started = time.time()
        try:
            r = process(SELF_CAMERA_ID, grab())
            print(f"[{r['time']}] p={r['probability']:.2f} "
                  f"alert={'YES' if r['alert'] else 'no'}"
                  f"{'  alert #' + str(r['alert_id']) if r['alert_id'] else ''}", flush=True)
        except Exception as e:  # keep running; one bad frame must not stop monitoring
            print(f"capture error: {e}", flush=True)
        stop.wait(max(0.0, interval - (time.time() - started)))


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class AckBody(BaseModel):
    by: Optional[str] = None


class ResolveBody(BaseModel):
    reason: str = "assisted"
    by: Optional[str] = None


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app(predictor=None, engine: Optional[AlertEngine] = None,
               manager: Optional[AlertManager] = None,
               capture_interval: Optional[float] = None,
               camera_factory: Optional[Callable[[], Callable[[], bytes]]] = None,
               api_key: Optional[str] = None) -> FastAPI:

    key = api_key if api_key is not None else (os.environ.get("ED_API_KEY") or None)
    ctx: Dict[str, object] = {"predictor": predictor, "engine": engine, "manager": manager,
                              "latest": {}, "lock": threading.Lock(), "capturing": False,
                              "shutdown": threading.Event()}

    def check_key(request: Request):
        if not key:
            return
        given = request.headers.get("x-api-key") or request.query_params.get("key") or ""
        if not hmac.compare_digest(given.encode(), key.encode()):
            raise HTTPException(401, "missing or wrong API key")

    auth = [Depends(check_key)]

    def process(camera_id: str, data: bytes) -> dict:
        p = ctx["predictor"].predict(data)               # may raise ValueError (bad image)
        positive = p >= ctx["predictor"].threshold
        streak, alert, new_alert = ctx["engine"].update(camera_id, positive)
        alert_id = ctx["manager"].on_photo(camera_id, float(p), alert)
        result = {
            "camera_id": camera_id,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "alert": alert,                # the detector's yes/no for this moment
            "alert_id": alert_id,          # the tracked alert for this fall, if any
            "fall_in_photo": bool(positive),
            "probability": round(float(p), 4),
            "consecutive": streak,
            "needed": ctx["engine"].consecutive,
            "threshold": round(float(ctx["predictor"].threshold), 4),
            # Legacy: repeats every cooldown while the person stays down. Frontends
            # should use /events and alert_id instead.
            "new_alert": new_alert,
        }
        with ctx["lock"]:
            ctx["latest"][camera_id] = result
        return result

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if ctx["predictor"] is None:
            ctx["predictor"] = load_predictor(
                os.environ.get("ED_MODEL", "ed_model.onnx"), env_float("ED_THRESHOLD", None))
        if ctx["engine"] is None:
            ctx["engine"] = AlertEngine(
                consecutive=int(env_float("ED_CONSECUTIVE", 2)),
                cooldown_s=60.0, max_gap_s=env_float("ED_OFFLINE_AFTER", 30.0))
        if ctx["manager"] is None:
            ctx["manager"] = AlertManager(
                path=os.environ.get("ED_ALERTS_FILE", "alerts.json"),
                offline_after=env_float("ED_OFFLINE_AFTER", 30.0),
                remind_every=env_float("ED_REMIND_EVERY", 60.0),
                clear_after=int(env_float("ED_CLEAR_AFTER", 3)),
                webhook_url=os.environ.get("ED_WEBHOOK_URL") or None)

        interval = capture_interval if capture_interval is not None \
            else env_float("ED_CAPTURE_INTERVAL", 0.0)
        for cid in filter(None, (os.environ.get("ED_EXPECTED_CAMERAS") or "").split(",")):
            ctx["manager"].register_camera(cid.strip())
        if interval and interval > 0:
            ctx["manager"].register_camera(SELF_CAMERA_ID)

        ctx["manager"].start()
        stop = threading.Event()
        if interval and interval > 0:
            grab = (camera_factory or make_camera)()
            threading.Thread(target=capture_loop, daemon=True,
                             args=(grab, process, interval, stop)).start()
            ctx["capturing"] = True
            print(f"Capturing a photo every {interval}s", flush=True)

        print(f"Model loaded. Threshold {ctx['predictor'].threshold:.3f}, "
              f"{ctx['engine'].consecutive} consecutive photos needed.", flush=True)
        if not key:
            print("WARNING: ED_API_KEY is not set - the API is open to anyone on the "
                  "network.", flush=True)
        yield
        ctx["shutdown"].set()
        stop.set()
        ctx["manager"].stop()

    app = FastAPI(title="ED fall detection", lifespan=lifespan)
    origins = [o.strip() for o in os.environ.get("ED_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["*"], allow_headers=["*"])

    # ---- photos in --------------------------------------------------------

    @app.post("/predict", dependencies=auth)
    async def predict(request: Request, camera_id: str = "default", format: str = "json"):
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = next((v for v in form.values() if hasattr(v, "read")), None)
            if upload is None:
                raise HTTPException(400, "multipart request contained no file")
            data = await upload.read()
        else:
            data = await request.body()
        if not data:
            raise HTTPException(400, "empty request body; send a JPEG image")
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "image too large (max 10 MB)")
        try:
            result = await run_in_threadpool(process, camera_id, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        if format == "text":
            return PlainTextResponse("yes" if result["alert"] else "no")
        return result

    # ---- alerts out -------------------------------------------------------

    @app.get("/events", dependencies=auth)
    async def events(request: Request):
        hub = ctx["manager"].hub

        async def stream():
            sub = hub.subscribe()
            q = sub[1]
            try:
                yield sse("snapshot", ctx["manager"].snapshot())
                last_beat = time.monotonic()
                while not ctx["shutdown"].is_set():
                    if await request.is_disconnected():
                        break
                    try:
                        event, data = await asyncio.wait_for(q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if time.monotonic() - last_beat >= HEARTBEAT_S:
                            yield sse("heartbeat", {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                            last_beat = time.monotonic()
                        continue
                    yield sse(event, data)
            finally:
                hub.unsubscribe(sub)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/alerts", dependencies=auth)
    def list_alerts(status: Optional[str] = None, limit: int = 50):
        if status not in (None, "open", "active", "acknowledged", "resolved"):
            raise HTTPException(400, "status must be open, active, acknowledged or resolved")
        return ctx["manager"].list_alerts(status, max(1, min(limit, 500)))

    @app.get("/alerts/{alert_id}", dependencies=auth)
    def get_alert(alert_id: int):
        a = ctx["manager"].get(alert_id)
        if a is None:
            raise HTTPException(404, "no such alert")
        return a

    @app.post("/alerts/{alert_id}/acknowledge", dependencies=auth)
    def acknowledge(alert_id: int, body: Optional[AckBody] = None):
        try:
            return ctx["manager"].acknowledge(alert_id, body.by if body else None)
        except KeyError:
            raise HTTPException(404, "no such alert")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.post("/alerts/{alert_id}/resolve", dependencies=auth)
    def resolve(alert_id: int, body: Optional[ResolveBody] = None):
        body = body or ResolveBody()
        try:
            return ctx["manager"].resolve(alert_id, body.reason, body.by)
        except KeyError:
            raise HTTPException(404, "no such alert")
        except ValueError as e:
            code = 409 if "already" in str(e) else 400
            raise HTTPException(code, str(e))

    @app.get("/cameras", dependencies=auth)
    def cameras():
        return ctx["manager"].cameras()

    # ---- misc -------------------------------------------------------------

    @app.get("/status", dependencies=auth)
    def status():
        with ctx["lock"]:
            return dict(ctx["latest"])

    @app.get("/health")
    def health():
        m = ctx["manager"]
        return {"model_loaded": ctx["predictor"] is not None,
                "capturing": ctx["capturing"],
                "phones_connected": m.hub.count() if m else 0}

    @app.post("/reset", dependencies=auth)
    def reset(camera_id: Optional[str] = None):
        ctx["engine"].reset(camera_id)
        return {"ok": True}

    return app


app = create_app()