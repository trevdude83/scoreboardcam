from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    mode: str
    model_path: str
    labels_path: str
    template_dir: str
    template_threshold: float
    template_min_matches: int
    template_scales: List[float]
    scoreboard_label: str
    invert: bool
    threshold: float
    required_hits: int
    window_size: int
    cooldown_seconds: int
    auto_calibrate: bool
    calibrate_min_score: float
    calibrate_min_matches: int
    calibrate_margin_left: float
    calibrate_margin_right: float
    calibrate_margin_top: float
    calibrate_margin_bottom: float
    calibrate_use_frame_margins: bool
    calibrate_frame_margin_left: float
    calibrate_frame_margin_right: float
    calibrate_frame_margin_top: float
    calibrate_frame_margin_bottom: float


@dataclass
class UploadConfig:
    max_images: int
    jpeg_quality: int
    format: str
    process_after_upload: bool


@dataclass
class DatasetConfig:
    path: str
    sample_fps: int
    burst_fps: int
    burst_frames: int
    save_full_frame: bool
    full_frame_every_n: int
    format: str


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
    dataset: DatasetConfig
    polling: PollingConfig
    logging: LoggingConfig
    local_server: LocalServerConfig
    spool: SpoolConfig


def _crop_override_path(path: str) -> Path:
    config_path = Path(path)
    if config_path.suffix.lower() == ".yaml":
        return config_path.with_name(f"{config_path.stem}.crop.yaml")
    if config_path.suffix.lower() == ".yml":
        return config_path.with_name(f"{config_path.stem}.crop.yml")
    return config_path.with_name(f"{config_path.name}.crop.yaml")


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    override_path = _crop_override_path(path)
    if override_path.exists():
        override = yaml.safe_load(override_path.read_text()) or {}
        camera = data.get("camera", {})
        crop = camera.get("crop", {})
        override_camera = override.get("camera", {})
        override_crop = override_camera.get("crop", {})
        if isinstance(override_crop, dict):
            crop.update(override_crop)
            camera["crop"] = crop
            data["camera"] = camera
    return _parse_config(data)


def crop_override_path(path: str = "config.yaml") -> Path:
    return _crop_override_path(path)


def _parse_config(data: Dict[str, Any]) -> AppConfig:
    server = data.get("server", {})
    device = data.get("device", {})
    camera = data.get("camera", {})
    detector = data.get("detector", {})
    upload = data.get("upload", {})
    dataset = data.get("dataset", {})
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

    template_scales: List[float] = [1.0]
    raw_scales = detector.get("templateScales")
    if isinstance(raw_scales, list):
        template_scales = [float(value) for value in raw_scales if value is not None]
    elif isinstance(raw_scales, str):
        parsed = []
        for part in raw_scales.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(float(part))
            except ValueError:
                continue
        if parsed:
            template_scales = parsed

    detector_cfg = DetectorConfig(
        enabled=bool(detector.get("enabled", False)),
        mode=str(detector.get("mode", "tflite")),
        model_path=str(detector.get("modelPath", "models/scoreboard_detector.tflite")),
        labels_path=str(detector.get("labelsPath", "models/labels.txt")),
        template_dir=str(detector.get("templateDir", "models/templates")),
        template_threshold=float(detector.get("templateThreshold", 0.8)),
        template_min_matches=int(detector.get("templateMinMatches", 3)),
        template_scales=template_scales,
        scoreboard_label=str(detector.get("scoreboardLabel", "scoreboard")),
        invert=bool(detector.get("invert", False)),
        threshold=float(detector.get("threshold", 0.8)),
        required_hits=int(detector.get("requiredHits", 8)),
        window_size=int(detector.get("windowSize", 10)),
        cooldown_seconds=int(detector.get("cooldownSeconds", 75)),
        auto_calibrate=bool(detector.get("autoCalibrate", True)),
        calibrate_min_score=float(detector.get("calibrateMinScore", 0.45)),
        calibrate_min_matches=int(detector.get("calibrateMinMatches", 3)),
        calibrate_margin_left=float(detector.get("calibrateMarginLeft", 1.4)),
        calibrate_margin_right=float(detector.get("calibrateMarginRight", 0.4)),
        calibrate_margin_top=float(detector.get("calibrateMarginTop", 0.6)),
        calibrate_margin_bottom=float(detector.get("calibrateMarginBottom", 3.2)),
        calibrate_use_frame_margins=bool(detector.get("calibrateUseFrameMargins", True)),
        calibrate_frame_margin_left=float(detector.get("calibrateFrameMarginLeft", 0.55)),
        calibrate_frame_margin_right=float(detector.get("calibrateFrameMarginRight", 0.05)),
        calibrate_frame_margin_top=float(detector.get("calibrateFrameMarginTop", 0.05)),
        calibrate_frame_margin_bottom=float(detector.get("calibrateFrameMarginBottom", 0.35)),
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

    dataset_format = str(dataset.get("format", "png")).lower()
    if dataset_format not in ("jpg", "jpeg", "png"):
        dataset_format = "png"

    dataset_cfg = DatasetConfig(
        path=str(dataset.get("path", "dataset")),
        sample_fps=int(dataset.get("sampleFps", 1)),
        burst_fps=int(dataset.get("burstFps", 8)),
        burst_frames=int(dataset.get("burstFrames", 12)),
        save_full_frame=bool(dataset.get("saveFullFrame", True)),
        full_frame_every_n=int(dataset.get("fullFrameEveryN", 10)),
        format=dataset_format,
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
        dataset=dataset_cfg,
        polling=polling_cfg,
        logging=logging_cfg,
        local_server=local_server_cfg,
        spool=spool_cfg,
    )


def require_config(value: str, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required in the config file")
    return value
