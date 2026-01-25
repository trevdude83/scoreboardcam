from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .camera import Camera
import cv2
from .client import DeviceClient
from .config import AppConfig, load_config, require_config
from .debounce import DebouncedWindow
from .detector import ScoreboardDetector
from .local_server import start_local_server
from .logging_utils import setup_logging
from .spool import delete_spool_item, list_spool, save_to_spool


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextState:
    def __init__(self) -> None:
        self.session_id: Optional[int] = None
        self.last_context_log = 0.0
        self.lock = threading.Lock()

    def update(self, session_id: Optional[int]) -> None:
        with self.lock:
            self.session_id = session_id

    def get(self) -> Optional[int]:
        with self.lock:
            return self.session_id


def poll_context(client: DeviceClient, state: ContextState, interval: int) -> None:
    while True:
        try:
            context = client.get_context()
            state.update(context.session_id)
            logging.info("Context: activeSession=%s", context.session_id)
        except Exception as exc:
            logging.warning("Context poll failed: %s", exc)
        time.sleep(interval)


def upload_images(
    client: DeviceClient,
    config: AppConfig,
    images: List[bytes],
    session_id: Optional[int],
    file_ext: str,
    mime_type: str,
) -> None:
    captured_at = iso_now()
    response = client.upload_images(images, session_id, captured_at, file_ext, mime_type)
    ingest_id = response.get("ingestId")
    status = response.get("status")
    logging.info("Upload complete: ingestId=%s status=%s", ingest_id, status)
    if config.upload.process_after_upload and isinstance(ingest_id, int):
        try:
            processed = client.process_ingest(ingest_id)
            logging.info("Process result: %s", processed)
        except Exception as exc:
            logging.warning("Process request failed: %s", exc)


def capture_and_upload(camera: Camera, client: DeviceClient, config: AppConfig, session_id: Optional[int]) -> None:
    image_bytes = b""
    try:
        image_bytes, ext, mime = camera.capture_image(config.upload.format, config.upload.jpeg_quality)
        upload_images(client, config, [image_bytes], session_id, ext, mime)
    except Exception as exc:
        logging.error("Upload failed, spooling: %s", exc)
        if image_bytes:
            save_to_spool(
                Path(config.spool.path),
                image_bytes,
                {
                    "capturedAt": iso_now(),
                    "sessionId": session_id,
                    "format": config.upload.format,
                    "error": str(exc),
                },
            )


def save_dataset_item(
    base_dir: Path,
    kind: str,
    index: int,
    image_bytes: bytes,
    ext: str,
    metadata: dict,
    suffix: str,
) -> None:
    date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_dir = base_dir / date_folder / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{timestamp}_{kind}_{index}_{suffix}"
    image_path = target_dir / f"{stem}.{ext}"
    meta_path = target_dir / f"{stem}.json"
    image_path.write_bytes(image_bytes)
    meta_path.write_text(json.dumps(metadata, indent=2))


def capture_dataset_sample(
    camera: Camera,
    config: AppConfig,
    base_dir: Path,
    index: int,
    mode: str,
    burst_index: Optional[int] = None,
) -> None:
    dataset_format = config.dataset.format
    crop_meta = asdict(config.camera.crop)
    meta = {
        "capturedAt": iso_now(),
        "mode": mode,
        "index": index,
        "burstIndex": burst_index,
        "crop": crop_meta,
        "camera": {
            "width": config.camera.width,
            "height": config.camera.height,
            "fps": config.camera.fps,
            "format": config.camera.format,
        },
        "format": dataset_format,
    }

    image_bytes, ext, _ = camera.capture_image(dataset_format, config.upload.jpeg_quality)
    save_dataset_item(base_dir, mode, index, image_bytes, ext, meta, "roi")

    if config.dataset.save_full_frame:
        interval = max(1, config.dataset.full_frame_every_n)
        if index % interval == 0:
            full_bytes, full_ext, _ = camera.capture_image(dataset_format, config.upload.jpeg_quality, apply_crop=False)
            full_meta = {**meta, "fullFrame": True}
            save_dataset_item(base_dir, mode, index, full_bytes, full_ext, full_meta, "full")


def run_collect(config: AppConfig) -> None:
    camera = Camera(config.camera)
    base_dir = Path(config.dataset.path)
    sample_interval = 1.0 / max(1, config.dataset.sample_fps)
    burst_interval = 1.0 / max(1, config.dataset.burst_fps)
    burst_frames = max(1, config.dataset.burst_frames)

    command_queue: List[str] = []
    queue_lock = threading.Lock()

    def input_loop() -> None:
        while True:
            command = input().strip().lower()
            with queue_lock:
                command_queue.append(command)

    input_thread = threading.Thread(target=input_loop, daemon=True)
    input_thread.start()

    logging.info("Dataset capture started. Press 'b' then ENTER for a burst, 'q' to quit.")

    index = 0
    try:
        while True:
            now = time.time()
            capture_dataset_sample(camera, config, base_dir, index, "sample")
            index += 1

            with queue_lock:
                commands = command_queue[:]
                command_queue.clear()

            for cmd in commands:
                if cmd in ("q", "quit", "exit"):
                    logging.info("Stopping dataset capture.")
                    return
                if cmd in ("b", "burst"):
                    logging.info("Burst capture triggered.")
                    for burst_index in range(burst_frames):
                        capture_dataset_sample(camera, config, base_dir, index, "burst", burst_index=burst_index)
                        index += 1
                        time.sleep(burst_interval)

            elapsed = time.time() - now
            sleep_for = sample_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        camera.release()


def run_manual_capture(config: AppConfig) -> None:
    client = DeviceClient(config.server, require_device(config))
    camera = Camera(config.camera)
    try:
        context = client.get_context()
        session_id = context.session_id
        capture_and_upload(camera, client, config, session_id)
    finally:
        camera.release()


def run_flush_spool(config: AppConfig) -> None:
    client = DeviceClient(config.server, require_device(config))
    spool_dir = Path(config.spool.path)
    items = list_spool(spool_dir)
    if not items:
        logging.info("No spool items to replay.")
        return
    for item in items:
        session_id = item.metadata.get("sessionId")
        session_id_value = int(session_id) if isinstance(session_id, int) else None
        try:
            ext = str(item.metadata.get("format") or "jpg")
            mime = "image/png" if ext == "png" else "image/jpeg"
            upload_images(client, config, [item.image_path.read_bytes()], session_id_value, ext, mime)
            delete_spool_item(item)
        except Exception as exc:
            logging.warning("Spool item failed: %s", exc)


def run_continuous(config: AppConfig, config_path: str) -> None:
    client = DeviceClient(config.server, require_device(config))
    camera = Camera(config.camera)
    state = ContextState()
    camera_lock = threading.Lock()

    poll_thread = threading.Thread(
        target=poll_context,
        args=(client, state, config.polling.context_seconds),
        daemon=True,
    )
    poll_thread.start()

    detector: ScoreboardDetector | None = None
    if config.detector.enabled:
        detector = ScoreboardDetector(
            config.detector.model_path,
            config.detector.labels_path,
            invert=config.detector.invert,
        )

    if config.local_server.enabled:
        preview_meta = {
            "width": config.camera.width,
            "height": config.camera.height,
            "crop": asdict(config.camera.crop),
        }
        def safe_capture() -> None:
            with camera_lock:
                capture_and_upload(camera, client, config, state.get())

        def safe_preview() -> bytes:
            with camera_lock:
                return camera.capture_image("jpg", config.upload.jpeg_quality, apply_crop=False)[0]

        def update_crop(data: dict) -> dict:
            crop = config.camera.crop
            if "enabled" in data:
                crop.enabled = bool(data.get("enabled"))
            for key in ("x", "y", "w", "h"):
                if key in data:
                    try:
                        setattr(crop, key, int(data.get(key)))
                    except (TypeError, ValueError):
                        continue
            write_crop_to_config(config_path, crop)
            return asdict(crop)

        def probe_detector(count: int, delay_seconds: float) -> dict:
            if detector is None:
                return {"error": "Detector is disabled."}
            results: list[float] = []
            for _ in range(count):
                with camera_lock:
                    frame = camera.read().image
                result = detector.classify(frame)
                scoreboard_label = config.detector.scoreboard_label
                if result.label == scoreboard_label:
                    scoreboard_prob = result.confidence
                else:
                    scoreboard_prob = 1.0 - result.confidence
                results.append(scoreboard_prob)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
            if not results:
                return {"count": 0}
            return {
                "count": len(results),
                "min": min(results),
                "max": max(results),
                "avg": sum(results) / len(results),
                "threshold": config.detector.threshold,
            }

        start_local_server(
            config.local_server.port,
            safe_capture,
            safe_preview,
            preview_meta,
            update_crop,
            probe_callback=probe_detector,
        )

    if config.detector.enabled:
        debounce = DebouncedWindow(
            scoreboard_label=config.detector.scoreboard_label,
            threshold=config.detector.threshold,
            required_hits=config.detector.required_hits,
            window_size=config.detector.window_size,
            cooldown_seconds=config.detector.cooldown_seconds,
        )
        frame_buffer: List[bytes] = []
        results_buffer: List[float] = []
        last_detector_log = 0.0
        while True:
            with camera_lock:
                frame = camera.read().image
            result = detector.classify(frame)
            decision = debounce.update(result)
            frame_bytes = encode_frame(frame, config.upload.format, config.upload.jpeg_quality)
            frame_buffer.append(frame_bytes)
            results_buffer.append(result.confidence)
            if len(frame_buffer) > config.detector.window_size:
                frame_buffer.pop(0)
                results_buffer.pop(0)
            now = time.time()
            if now - last_detector_log >= 2.0:
                logging.info(
                    "Detector: label=%s confidence=%.3f threshold=%.3f",
                    result.label,
                    result.confidence,
                    config.detector.threshold,
                )
                last_detector_log = now
            if decision.triggered and decision.best_index is not None:
                best_index = decision.best_index
                best_frame = frame_buffer[best_index]
                images = select_best_frames(frame_buffer, results_buffer, config.upload.max_images)
                ext = "png" if config.upload.format == "png" else "jpg"
                mime = "image/png" if ext == "png" else "image/jpeg"
                upload_images(client, config, images, state.get(), ext, mime)
            time.sleep(1.0 / max(1, config.camera.fps))
    else:
        logging.info("Detector disabled. Press ENTER to capture, or use /capture endpoint.")
        try:
            while True:
                input()
                capture_and_upload(camera, client, config, state.get())
        except KeyboardInterrupt:
            logging.info("Shutting down.")
        finally:
            camera.release()


def require_device(config: AppConfig):
    require_config(config.device.device_id, "device.deviceId")
    require_config(config.device.device_key, "device.deviceKey")
    return config.device


def write_crop_to_config(config_path: str, crop) -> None:
    try:
        import yaml
        from pathlib import Path
        config_file = Path(config_path)
        data = yaml.safe_load(config_file.read_text()) or {}
        camera = data.get("camera", {})
        camera["crop"] = {
            "enabled": bool(crop.enabled),
            "x": int(crop.x),
            "y": int(crop.y),
            "w": int(crop.w),
            "h": int(crop.h),
        }
        data["camera"] = camera
        config_file.write_text(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        return


def select_best_frames(frames: List[bytes], confidences: List[float], max_images: int) -> List[bytes]:
    if not frames:
        return []
    indexed = list(enumerate(confidences))
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    picks = [frames[index] for index, _ in indexed[:max_images]]
    return picks


def encode_frame(frame, fmt: str, quality: int) -> bytes:
    fmt_lower = fmt.lower()
    if fmt_lower == "png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        ok, buffer = cv2.imencode(".png", frame, encode_params)
        if not ok:
            raise RuntimeError("Failed to encode PNG frame.")
        return buffer.tobytes()
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame.")
    return buffer.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="RocketSessions ScoreboardCam client")
    parser.add_argument("command", choices=["capture", "run", "flush-spool", "collect"])
    parser.add_argument("--config")
    args = parser.parse_args()

    config_path = args.config
    if not config_path:
        config_path = "config.local.yaml" if Path("config.local.yaml").exists() else "config.yaml"
    config = load_config(config_path)
    setup_logging(config.logging.level)

    if args.command == "capture":
        run_manual_capture(config)
    elif args.command == "flush-spool":
        run_flush_spool(config)
    elif args.command == "collect":
        run_collect(config)
    else:
        run_continuous(config, config_path)


if __name__ == "__main__":
    main()
