from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # pragma: no cover
    tflite = None


@dataclass
class DetectionResult:
    label: str
    confidence: float


class ScoreboardDetector:
    def __init__(self, model_path: str, labels_path: str) -> None:
        if tflite is None:
            raise RuntimeError("tflite-runtime is not installed.")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = _load_labels(labels_path)

    def classify(self, frame: np.ndarray) -> DetectionResult:
        input_data = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        if output_data.shape == () or output_data.shape == (1,):
            # Binary sigmoid output: value is P(scoreboard).
            scoreboard_prob = float(output_data.item())
            label = "scoreboard"
            if len(self.labels) >= 2:
                label = self.labels[1]
            if scoreboard_prob < 0.5:
                label = self.labels[0] if self.labels else "not_scoreboard"
            confidence = scoreboard_prob if label != "not_scoreboard" else 1.0 - scoreboard_prob
            return DetectionResult(label=label, confidence=confidence)
        best_index = int(np.argmax(output_data))
        confidence = float(output_data[best_index])
        label = self.labels[best_index] if best_index < len(self.labels) else str(best_index)
        return DetectionResult(label=label, confidence=confidence)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32)
        # Match tf.keras.applications.mobilenet_v3.preprocess_input
        input_data = (input_data / 127.5) - 1.0
        return np.expand_dims(input_data, axis=0)


def _load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]
