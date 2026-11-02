@echo off
REM One-click demo launcher (Windows). Double-click it, or run it from the project folder.
REM It starts the backend with all three alert types switched on:
REM   fall        - real detector, plus a printed fall picture (demo_refs\fall.jpg)
REM   no_movement - real detection: someone in view who has not moved for 20 seconds
REM   chest_pain  - DEMO: a printed chest-pain picture (demo_refs\chest_pain.jpg)
REM Old alerts are cleared each time so the demo starts clean.

cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate.bat

if exist ed_model.onnx (set ED_MODEL=ed_model.onnx) else (set ED_MODEL=ed_model.pt)
set ED_DEMO=1
set ED_STILL=1
set ED_STILL_AFTER=20
set ED_CAPTURE_INTERVAL=2
if exist alerts.json del alerts.json

echo.
echo  Starting demo (model: %ED_MODEL%). Leave this window open.
echo  1) With NOBODY in the camera view, open http://localhost:8000/docs
echo     and run  POST /stillness/calibrate  with camera_id = pi-camera
echo  2) Then sit still for 20 seconds, or hold up the printed pictures.
echo  Stop with Ctrl+C.
echo.
python -m uvicorn server:app --port 8000 --timeout-graceful-shutdown 3
pause
