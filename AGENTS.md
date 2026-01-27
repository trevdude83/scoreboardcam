# AGENTS.md

## Project
ScoreboardCam (Raspberry Pi client) for RocketSessions Scoreboard ingestion.

## Run
- Primary: `python -m src.main run --config config.yaml`
- One-off capture: `python -m src.main capture --config config.yaml`
- Preview: http://<pi-ip>:5055/preview.html

## Config
- Use `config.yaml` (not committed) for deviceId/deviceKey, camera, crop.
- Copy from `config.example.yaml` first.
- Auto-calibrate and preview crop updates write to `config.crop.yaml`.
- Detector mode in config:
  - template mode uses `templateThreshold` + `templateMinMatches` and PNG templates in `models/templates/`.
  - tflite mode uses `threshold` + model in `models/`.

## Crop & Debug
- ROI must include header row (SCORE/GOALS/ASSISTS/SAVES/SHOTS/PING).
- To save ROI debug frame (with crop):
  ```
  python - <<'PY'
  import cv2
  from src.config import load_config
  from src.camera import Camera
  cfg = load_config('config.yaml')
  cam = Camera(cfg.camera)
  frame = cam.read().image
  cam.release()
  cv2.imwrite('roi_debug.png', frame)
  print('saved roi_debug.png', frame.shape)
  PY
  ```

## Templates
- Place header templates in `models/templates/*.PNG`.
- Template matching threshold typically 0.60–0.75; tune using `maxScore` logs.

## Notes
- Preview crop now uses actual frame size; draw crop in preview then save.
- If detection fails with `matches=0`, templates likely mismatch or ROI is off.
