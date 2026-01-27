from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    meta: dict | None = None


class ScoreboardDetector:
    def __init__(self, model_path: str, labels_path: str, invert: bool = False) -> None:
        if tflite is None:
            raise RuntimeError("tflite-runtime is not installed.")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = _load_labels(labels_path)
        self.invert = invert

    def classify(self, frame: np.ndarray) -> DetectionResult:
        input_data = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        if output_data.shape == () or output_data.shape == (1,):
            # Binary sigmoid output: value is P(scoreboard).
            scoreboard_prob = float(output_data.item())
            if self.invert:
                scoreboard_prob = 1.0 - scoreboard_prob
            score_label = "scoreboard"
            not_label = "not_scoreboard"
            if len(self.labels) >= 2:
                not_label = self.labels[0]
                score_label = self.labels[1]
            label = score_label if scoreboard_prob >= 0.5 else not_label
            confidence = scoreboard_prob if label == score_label else 1.0 - scoreboard_prob
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


class TemplateDetector:
    def __init__(
        self,
        template_dir: str,
        threshold: float,
        min_matches: int,
        scoreboard_label: str = "scoreboard",
        scales: List[float] | None = None,
    ) -> None:
        self.threshold = threshold
        self.min_matches = min_matches
        self.scoreboard_label = scoreboard_label
        self.not_scoreboard_label = "not_scoreboard"
        self.scales = scales or [1.0]
        self.templates = self._load_templates(template_dir)
        if not self.templates:
            raise RuntimeError(f"No templates found in {template_dir}")

    def classify(self, frame: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        matches = 0
        max_score = 0.0
        best_scale = 1.0
        for template in self.templates:
            for scale in self.scales:
                if scale <= 0:
                    continue
                if scale == 1.0:
                    scaled = template
                else:
                    scaled = cv2.resize(
                        template,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_LINEAR,
                    )
                if gray.shape[0] < scaled.shape[0] or gray.shape[1] < scaled.shape[1]:
                    continue
                result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                score = float(score)
                if score > max_score:
                    max_score = score
                    best_scale = scale
                if score >= self.threshold:
                    matches += 1
                    break
        if matches >= self.min_matches:
            return DetectionResult(
                label=self.scoreboard_label,
                confidence=max_score,
                meta={"matches": matches, "maxScore": max_score, "bestScale": best_scale},
            )
        return DetectionResult(
            label=self.not_scoreboard_label,
            confidence=max_score,
            meta={"matches": matches, "maxScore": max_score, "bestScale": best_scale},
        )

    def _load_templates(self, template_dir: str) -> List[np.ndarray]:
        path = Path(template_dir)
        if not path.exists():
            return []
        templates: List[np.ndarray] = []
        for item in path.iterdir():
            if item.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            image = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            templates.append(image)
        return templates


def _load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]
