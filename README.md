# Early recognition of collapse in ED waiting rooms (proof of concept)

A small on-device image classifier that detects a person lying on the floor
(a collapse or fall) from a single still photo, plus a backend that turns
detections into alerts for a nurse's phone on the same local network.
Student project (ENG40011). **Research prototype. Not a medical device and not
for clinical use.**

## How it works

```
camera (Raspberry Pi)  ->  model (ONNX, runs on the Pi)  ->  alert manager  ->  phone (live feed)
   photo every ~5 s        photo -> fall probability          one alert per fall,
                                                              reminders, camera-offline
```

- Photos are processed in memory and never saved or sent anywhere.
- An alert needs 2 consecutive positive photos, which reduces one-off false alarms.
- If a camera stops reporting, that raises its own alert instead of looking "all clear".

## Results (held-out people, never seen in training)

Trained on the public CAUCAFall dataset (subjects 1 to 8), tested on subjects 9 and 10.

| AUROC | Sensitivity | Specificity |
|---|---|---|
| 0.956 | 0.95 | 0.87 |

Limitations: only two test people; one fixed corner camera in a home-style room;
the model detects the *lying-on-the-floor* state after a fall, not the fall motion;
about 13% of normal photos were flagged at the chosen threshold. It has not been
tested in a real waiting room. See the project report for discussion.

## Files

| File | Purpose |
|---|---|
| `make_dataset.py` | Build the training set from the CAUCAFall frame folders |
| `ed_model.py` | Train, evaluate and run the model (PyTorch) |
| `export_onnx.py` | Convert `ed_model.pt` to `ed_model.onnx` for the Pi |
| `check_onnx.py` | Confirm the ONNX model scores like the original |
| `fall_core.py` | Inference (ONNX or PyTorch) and the consecutive-photo rule |
| `alerting.py` | Alert lifecycle, reminders, camera health, live event hub |
| `server.py` | HTTP backend (FastAPI): photos in, alerts out |
| `send_test.py` | Send test photos to the server |
| `FRONTEND_API.md` | Guide for the phone app developer |

## Setup

PC (training and testing):
```
pip install -r requirements-train.txt
```

1. Download CAUCAFall (see below) into `Dataset CAUCAFall/`.
2. `python make_dataset.py inspect --root "Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall"`
3. `python make_dataset.py build --root "Dataset CAUCAFall/Dataset CAUCAFall/CAUCAFall" --test-subjects 9,10`
4. `python ed_model.py train --data data --epochs 15 --out ed_model.pt --workers 0`
5. `python ed_model.py eval ed_model.pt --data data_test --workers 0`
6. `python export_onnx.py ed_model.pt`

Run the backend on a PC:
```
ED_MODEL=ed_model.pt uvicorn server:app --port 8000 --timeout-graceful-shutdown 3
python send_test.py --folder data_test/clips --match Fallsitting --limit 40
```
(PowerShell: `$env:ED_MODEL="ed_model.pt"` before starting.)

Raspberry Pi: install `requirements-pi.txt`, copy `ed_model.onnx`, `ed_model.json`,
`server.py`, `alerting.py` and `fall_core.py`, then:
```
ED_API_KEY=choose-a-secret ED_CAPTURE_INTERVAL=5 \
uvicorn server:app --host 0.0.0.0 --port 8000 --timeout-graceful-shutdown 3
```
Settings are environment variables; see the top of `server.py`. **Never commit
the API key.**

## Dataset

The raw CAUCAFall download is not included in this repository (it is many GB).
The processed training photos (`data/`) and held-out test photos (`data_test/`)
built from it are included, so you can train and test straight after cloning.
To rebuild them or change the settings, download the dataset from
https://data.mendeley.com/datasets/7w7fccy7ky/4 (CC BY 4.0).

Eraso, J. C., Muñoz, E., Muñoz, M., and Pinto, J., "Dataset CAUCAFall,"
Mendeley Data, V4, 2022, doi: 10.17632/7w7fccy7ky.4