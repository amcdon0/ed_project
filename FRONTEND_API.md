# Fall alert backend: guide for the frontend

The backend runs on the Raspberry Pi. The phone and the Pi must be on the same network.
Base URL: `http://<pi-ip>:8000` (find the Pi's address with `hostname -I`).

## Authentication

If the Pi was started with `ED_API_KEY=...`, every endpoint except `/health` needs the key:

- Header: `X-API-Key: <key>` (use this for `fetch`), or
- Query string: `?key=<key>` (use this for `EventSource`, which can't set headers).

Wrong or missing key returns `401`.

## 1. Live feed: `GET /events` (Server-Sent Events)

This is the main feed. Open it once and keep it open.

```js
const API = "http://192.168.1.50:8000";
const KEY = "choose-a-secret";

const es = new EventSource(`${API}/events?key=${KEY}`);

es.addEventListener("snapshot", e => {
  const { alerts, cameras } = JSON.parse(e.data);   // replace your whole state with this
});
es.addEventListener("alert", e => {
  const alert = JSON.parse(e.data);                  // NEW alert: show full-screen, buzz, play sound
});
es.addEventListener("reminder", e => {
  const alert = JSON.parse(e.data);                  // still unacknowledged: buzz again
});
es.addEventListener("alert_update", e => {
  const alert = JSON.parse(e.data);                  // status changed: update it in your list
});
es.addEventListener("camera", e => {
  const camera = JSON.parse(e.data);                 // camera went offline or came back
});
es.addEventListener("heartbeat", e => { lastBeat = Date.now(); });
es.onerror = () => { /* shows "connection lost"; EventSource retries automatically */ };
```

Events:

| Event | When | What to do |
|---|---|---|
| `snapshot` | Right after connecting or reconnecting | Replace local state: `{alerts: [...open alerts], cameras: [...], server_time}` |
| `alert` | A new alert was created | Notify the nurse |
| `reminder` | An alert is still unacknowledged (every 60 s) | Notify again |
| `alert_update` | Acknowledged, resolved, or fall no longer in view | Update that alert in the UI |
| `camera` | A camera went offline or came back | Update the camera status |
| `heartbeat` | Every 15 s | If none arrives for ~40 s, show "connection lost" |

On every reconnect the server sends a fresh `snapshot`, so the UI is always correct after a dropped connection.

## 2. Alert object

```json
{
  "id": 1,
  "type": "fall",
  "camera_id": "pi-camera",
  "status": "active",
  "created_at": "2026-08-29T10:40:00+0000",
  "acknowledged_at": null,
  "acknowledged_by": null,
  "resolved_at": null,
  "resolved_by": null,
  "resolution": null,
  "probability": 0.93,
  "peak_probability": 0.93,
  "still_detected": true,
  "reminders": 0
}
```

- `type`: `"fall"` or `"camera_offline"`. Treat a camera going silent as seriously as a fall, because nobody is being watched.
- `status`: `active` (nobody has responded), `acknowledged` (a nurse is on it), `resolved` (closed).
- `still_detected`: `false` means the camera no longer sees the fall, for example the person was helped up. The alert stays open until a nurse resolves it.
- `resolution`: `assisted`, `false_alarm`, `other`, or `camera_back` (set automatically when an offline camera returns).
- `probability` is the model's confidence for the latest photo. Don't show it to nurses as a percentage of certainty, because it isn't calibrated.

## 3. Nurse actions

```js
// "I'm going"
await fetch(`${API}/alerts/${id}/acknowledge`, {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ by: "Nurse A" })          // body is optional
});

// "Done"
await fetch(`${API}/alerts/${id}/resolve`, {
  method: "POST",
  headers: { "X-API-Key": KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ reason: "assisted", by: "Nurse A" })   // or "false_alarm" / "other"
});
```

- Acknowledging stops the reminders.
- Resolving as `false_alarm` is useful data for the project: it records which alerts were wrong.
- After a nurse resolves an alert, no new alert is raised while the same person is still in view. A new alert appears only after the scene has been clear for a few photos.
- Errors: `404` unknown id, `409` already resolved, `400` invalid reason.

## 4. Other endpoints

| Endpoint | Returns |
|---|---|
| `GET /alerts?status=open` | Alerts, newest first. `status` can be `open` (active + acknowledged), `active`, `acknowledged`, `resolved`. Also `limit=50`. |
| `GET /alerts/{id}` | One alert |
| `GET /cameras` | `[{camera_id, online, last_seen, seconds_since_last_photo, last_probability, open_alert_id}]` |
| `GET /health` | `{model_loaded, capturing, phones_connected}`. No key needed. |

Use these to load history or as a fallback if `/events` isn't available.

## Things to know

- **Browsers can't buzz a locked phone** from a web page. This feed works while the page is open in the foreground. For alerts on a locked phone, the Pi can also send each alert to a webhook (`ED_WEBHOOK_URL`) for a push service such as ntfy or Pushover, which needs internet access.
- **Keep the page on plain `http://`** if the Pi is on `http://`. A page served over `https://` is blocked from calling an `http://` Pi.
- **No photos** are ever sent to the frontend. Only alert data (camera, time, confidence).
- Cross-origin calls are allowed (`ED_CORS_ORIGINS`, default `*`), so the page can be served from anywhere on the network.
- Only one nurse needs to acknowledge. All connected phones see the `alert_update` immediately.
