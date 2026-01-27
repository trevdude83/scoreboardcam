# Scoreboard detector training (binary classifier)

This folder contains a simple training pipeline for the ScoreboardCam detector.

## Dataset layout

Place your ROI images in this structure:

```
dataset_clean/
  scoreboard/        # positives
  not_scoreboard/    # negatives
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```
python train_scoreboard.py --data-dir dataset_clean --epochs 10
```

This writes a SavedModel to `scoreboard_saved_model/`.

## Convert to TFLite

```
python convert_to_tflite.py --saved-model scoreboard_saved_model --output scoreboard_detector.tflite
```

## Labels file

Create a `labels.txt` file with:

```
not_scoreboard
scoreboard
```

## Deploy to Pi

Copy:
- `scoreboard_detector.tflite` -> `models/scoreboard_detector.tflite`
- `labels.txt` -> `models/labels.txt`

Enable in `config.yaml` (local file):

```
detector:
  enabled: true
```
