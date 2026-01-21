# rocketsessions-scoreboardcam

Raspberry Pi client for RocketSessions ScoreboardCam. Captures Rocket League end-of-match scoreboards from a USB webcam and uploads them to the RocketSessions server.

## Features

Phase 1 (enabled by default):
- Manual capture and upload via CLI.
- Poll server for active session context.
- Upload scoreboard images (1-3 frames).
- Optional local HTTP endpoint for remote trigger.
- Spool failed uploads for later replay.

Phase 2 (disabled by default):
- On-device scoreboard detection via TFLite MobileNet.
- Debounced detection with cooldown.
- Frame buffer and best-frame selection.

## Quick start

1) Create a device key on the RocketSessions server:

```
POST http://<server-host>:<port>/api/v1/scoreboard/devices/register
```

2) Copy the deviceId and deviceKey into `config.yaml`.

3) Install dependencies:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4) Run a manual capture:

```
python -m src.main capture
```

5) Run continuous mode:

```
python -m src.main run
```

Open a live preview in your browser (for camera positioning/crop):

```
http://<pi-ip>:5055/preview
```

Auto-refreshing preview page:

```
http://<pi-ip>:5055/preview.html
```

## Configuration

Edit `config.yaml`:

```
server:
  baseUrl: "http://<rocketsessions-host>:<port>"
device:
  deviceId: ""
  deviceKey: ""
  authHeaderMode: "x-device-key"  # or bearer
camera:
  index: 0
  width: 1280
  height: 720
  fps: 15
  format: "MJPG"
  controls:
    auto_exposure: 1
    exposure_time_absolute: 400
    sharpness: 8
    contrast: 20
    brightness: -10
    saturation: 45
    gamma: 100
    white_balance_automatic: 1
    white_balance_temperature: 4500
    backlight_compensation: 0
  rotateDegrees: 0
  crop:
    enabled: false
    x: 0
    y: 0
    w: 1280
    h: 720
detector:
  enabled: false
  modelPath: "models/scoreboard_detector.tflite"
  labelsPath: "models/labels.txt"
  scoreboardLabel: "scoreboard_end_match"
  threshold: 0.80
  requiredHits: 8
  windowSize: 10
  cooldownSeconds: 75
upload:
  maxImages: 3
  jpegQuality: 85
  processAfterUpload: true
polling:
  contextSeconds: 10
logging:
  level: "INFO"
localServer:
  enabled: true
  port: 5055
spool:
  path: "spool"
```

## Camera calibration tips

- Use `rotateDegrees` if your camera is mounted sideways.
- Use `crop` to isolate the scoreboard region for better detection accuracy.
- Verify capture quality with `python -m src.main capture`.

## Spool replay

If uploads fail, images and metadata are stored in `spool/`. Replay them with:

```
python -m src.main flush-spool
```

## systemd service

Install the service (optional):

```
sudo bash install.sh
```

This will install and enable `scoreboardcam.service` which runs:

```
python -m src.main run
```

## Notes

- Headless operation only. No GUI required.
- Phase 2 detector requires a TFLite model and labels file; default config keeps it off.
