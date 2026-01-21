from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class CropConfig:
    enabled: bool
    x: int
    y: int
    w: int
    h: int


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    format: str
    controls: Dict[str, Any]
    rotate_degrees: int
    crop: CropConfig


@dataclass
class DetectorConfig:
    enabled: bool
    model_path: str
    labels_path: str
    scoreboard_label: str
    threshold: float
    required_hits: int
    window_size: int
    cooldown_seconds: int


@dataclass
class UploadConfig:
    max_images: int
    jpeg_quality: int
    format: str
    process_after_upload: bool


@dataclass
class PollingConfig:
    context_seconds: int


@dataclass
class LoggingConfig:
    level: str


@dataclass
class LocalServerConfig:
    enabled: bool
    port: int


@dataclass
class SpoolConfig:
    path: str


@dataclass
class DeviceConfig:
    device_id: str
    device_key: str
    auth_header_mode: str


@dataclass
class ServerConfig:
    base_url: str


@dataclass
class AppConfig:
    server: ServerConfig
    device: DeviceConfig
    camera: CameraConfig
    detector: DetectorConfig
    upload: UploadConfig
    polling: PollingConfig
    logging: LoggingConfig
    local_server: LocalServerConfig
    spool: SpoolConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    return _parse_config(data)


def _parse_config(data: Dict[str, Any]) -> AppConfig:
    server = data.get("server", {})
    device = data.get("device", {})
    camera = data.get("camera", {})
    detector = data.get("detector", {})
    upload = data.get("upload", {})
    polling = data.get("polling", {})
    logging = data.get("logging", {})
    local_server = data.get("localServer", {})
    spool = data.get("spool", {})

    crop = camera.get("crop", {})
    crop_cfg = CropConfig(
        enabled=bool(crop.get("enabled", False)),
        x=int(crop.get("x", 0)),
        y=int(crop.get("y", 0)),
        w=int(crop.get("w", camera.get("width", 1280))),
        h=int(crop.get("h", camera.get("height", 720))),
    )

    camera_cfg = CameraConfig(
        index=int(camera.get("index", 0)),
        width=int(camera.get("width", 1280)),
        height=int(camera.get("height", 720)),
        fps=int(camera.get("fps", 15)),
        format=str(camera.get("format", "MJPG")),
        controls=dict(camera.get("controls", {})),
        rotate_degrees=int(camera.get("rotateDegrees", 0)),
        crop=crop_cfg,
    )

    detector_cfg = DetectorConfig(
        enabled=bool(detector.get("enabled", False)),
        model_path=str(detector.get("modelPath", "models/scoreboard_detector.tflite")),
        labels_path=str(detector.get("labelsPath", "models/labels.txt")),
        scoreboard_label=str(detector.get("scoreboardLabel", "scoreboard_end_match")),
        threshold=float(detector.get("threshold", 0.8)),
        required_hits=int(detector.get("requiredHits", 8)),
        window_size=int(detector.get("windowSize", 10)),
        cooldown_seconds=int(detector.get("cooldownSeconds", 75)),
    )

    format_value = str(upload.get("format", "jpg")).lower()
    if format_value not in ("jpg", "jpeg", "png"):
        format_value = "jpg"

    upload_cfg = UploadConfig(
        max_images=int(upload.get("maxImages", 3)),
        jpeg_quality=int(upload.get("jpegQuality", 85)),
        format=format_value,
        process_after_upload=bool(upload.get("processAfterUpload", True)),
    )

    polling_cfg = PollingConfig(
        context_seconds=int(polling.get("contextSeconds", 10)),
    )

    logging_cfg = LoggingConfig(
        level=str(logging.get("level", "INFO")),
    )

    local_server_cfg = LocalServerConfig(
        enabled=bool(local_server.get("enabled", True)),
        port=int(local_server.get("port", 5055)),
    )

    spool_cfg = SpoolConfig(
        path=str(spool.get("path", "spool")),
    )

    device_cfg = DeviceConfig(
        device_id=str(device.get("deviceId", "")),
        device_key=str(device.get("deviceKey", "")),
        auth_header_mode=str(device.get("authHeaderMode", "x-device-key")),
    )

    server_cfg = ServerConfig(
        base_url=str(server.get("baseUrl", "http://localhost:3001")),
    )

    return AppConfig(
        server=server_cfg,
        device=device_cfg,
        camera=camera_cfg,
        detector=detector_cfg,
        upload=upload_cfg,
        polling=polling_cfg,
        logging=logging_cfg,
        local_server=local_server_cfg,
        spool=spool_cfg,
    )


def require_config(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required in config.yaml")
    return value
