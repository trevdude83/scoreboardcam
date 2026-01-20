from __future__ import annotations

import argparse
import logging
import threading
import time
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
) -> None:
    captured_at = iso_now()
    response = client.upload_images(images, session_id, captured_at)
    ingest_id = response.get("ingestId")
    status = response.get("status")
    logging.info("Upload complete: ingestId=%s status=%s", ingest_id, status)
    if config.upload.process_after_upload and isinstance(ingest_id, int):
        processed = client.process_ingest(ingest_id)
        logging.info("Process result: %s", processed)


def capture_and_upload(camera: Camera, client: DeviceClient, config: AppConfig, session_id: Optional[int]) -> None:
    image_bytes = b""
    try:
        image_bytes = camera.capture_jpeg(config.upload.jpeg_quality)
        upload_images(client, config, [image_bytes], session_id)
    except Exception as exc:
        logging.error("Upload failed, spooling: %s", exc)
        if image_bytes:
            save_to_spool(
                Path(config.spool.path),
                image_bytes,
                {"capturedAt": iso_now(), "sessionId": session_id, "error": str(exc)},
            )


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
            upload_images(client, config, [item.image_path.read_bytes()], session_id_value)
            delete_spool_item(item)
        except Exception as exc:
            logging.warning("Spool item failed: %s", exc)


def run_continuous(config: AppConfig) -> None:
    client = DeviceClient(config.server, require_device(config))
    camera = Camera(config.camera)
    state = ContextState()

    poll_thread = threading.Thread(
        target=poll_context,
        args=(client, state, config.polling.context_seconds),
        daemon=True,
    )
    poll_thread.start()

    if config.local_server.enabled:
        start_local_server(config.local_server.port, lambda: capture_and_upload(camera, client, config, state.get()))

    if config.detector.enabled:
        detector = ScoreboardDetector(config.detector.model_path, config.detector.labels_path)
        debounce = DebouncedWindow(
            scoreboard_label=config.detector.scoreboard_label,
            threshold=config.detector.threshold,
            required_hits=config.detector.required_hits,
            window_size=config.detector.window_size,
            cooldown_seconds=config.detector.cooldown_seconds,
        )
        frame_buffer: List[bytes] = []
        results_buffer: List[float] = []
        while True:
            frame = camera.read().image
            result = detector.classify(frame)
            decision = debounce.update(result)
            frame_bytes = encode_jpeg(frame, config.upload.jpeg_quality)
            frame_buffer.append(frame_bytes)
            results_buffer.append(result.confidence)
            if len(frame_buffer) > config.detector.window_size:
                frame_buffer.pop(0)
                results_buffer.pop(0)
            if decision.triggered and decision.best_index is not None:
                best_index = decision.best_index
                best_frame = frame_buffer[best_index]
                images = select_best_frames(frame_buffer, results_buffer, config.upload.max_images)
                upload_images(client, config, images, state.get())
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


def require_device(config: AppConfig) -> AppConfig:
    require_config(config.device.device_id, "device.deviceId")
    require_config(config.device.device_key, "device.deviceKey")
    return config


def select_best_frames(frames: List[bytes], confidences: List[float], max_images: int) -> List[bytes]:
    if not frames:
        return []
    indexed = list(enumerate(confidences))
    indexed.sort(key=lambda pair: pair[1], reverse=True)
    picks = [frames[index] for index, _ in indexed[:max_images]]
    return picks


def encode_jpeg(frame, quality: int) -> bytes:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame.")
    return buffer.tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="RocketSessions ScoreboardCam client")
    parser.add_argument("command", choices=["capture", "run", "flush-spool"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging.level)

    if args.command == "capture":
        run_manual_capture(config)
    elif args.command == "flush-spool":
        run_flush_spool(config)
    else:
        run_continuous(config)


if __name__ == "__main__":
    main()
