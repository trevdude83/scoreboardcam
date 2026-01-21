from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .config import CameraConfig


@dataclass
class CameraFrame:
    image: np.ndarray


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap = cv2.VideoCapture(config.index)
        if config.format:
            fourcc = cv2.VideoWriter_fourcc(*config.format)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)

    def read(self) -> CameraFrame:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        frame = self._apply_rotation(frame, self.config.rotate_degrees)
        frame = self._apply_crop(frame)
        return CameraFrame(image=frame)

    def capture_jpeg(self, quality: int = 85) -> bytes:
        frame = self.read().image
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
        ok, buffer = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode JPEG frame.")
        return buffer.tobytes()

    def release(self) -> None:
        if self.cap:
            self.cap.release()

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        crop = self.config.crop
        if not crop.enabled:
            return frame
        h, w = frame.shape[:2]
        x = max(0, min(crop.x, w - 1))
        y = max(0, min(crop.y, h - 1))
        cw = max(1, min(crop.w, w - x))
        ch = max(1, min(crop.h, h - y))
        return frame[y : y + ch, x : x + cw]

    def _apply_rotation(self, frame: np.ndarray, degrees: int) -> np.ndarray:
        normalized = degrees % 360
        if normalized == 0:
            return frame
        if normalized == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if normalized == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if normalized == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, normalized, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))
