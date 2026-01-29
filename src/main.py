from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from .camera import Camera
import cv2
from .client import DeviceClient
from .config import AppConfig, crop_override_path, load_config, require_config
from .debounce import DebouncedWindow
from .detector import ScoreboardDetector, TemplateDetector
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
    detector_lock = threading.Lock()

    auto_calibrate_crop(camera, config, config_path)

    poll_thread = threading.Thread(
        target=poll_context,
        args=(client, state, config.polling.context_seconds),
        daemon=True,
    )
    poll_thread.start()

    detector: ScoreboardDetector | TemplateDetector | None = None
    detector_mode = config.detector.mode.lower()
    if config.detector.enabled:
        if detector_mode == "template":
            detector = TemplateDetector(
                config.detector.template_dir,
                config.detector.template_threshold,
                config.detector.template_min_matches,
                scoreboard_label=config.detector.scoreboard_label,
                scales=config.detector.template_scales,
            )
        else:
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
                frame = camera.read_raw()
                preview_meta["width"] = frame.shape[1]
                preview_meta["height"] = frame.shape[0]
                return camera.encode_image(frame, "jpg", config.upload.jpeg_quality)[0]

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
                with detector_lock:
                    result = detector.classify(frame)
                if detector_mode == "template":
                    scoreboard_prob = result.confidence
                else:
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
        threshold = config.detector.threshold
        if detector_mode == "template":
            threshold = config.detector.template_threshold
        debounce = DebouncedWindow(
            scoreboard_label=config.detector.scoreboard_label,
            threshold=threshold,
            required_hits=config.detector.required_hits,
            window_size=config.detector.window_size,
            cooldown_seconds=config.detector.cooldown_seconds,
            rearm_min_clears=config.detector.rearm_min_clears,
        )
        frame_buffer: List[bytes] = []
        results_buffer: List[float] = []
        last_detector_log = 0.0
        while True:
            with camera_lock:
                frame = camera.read().image
            with detector_lock:
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
                extra = ""
                if result.meta:
                    extra = " matches={matches} maxScore={maxScore:.3f}".format(**result.meta)
                threshold_value = (
                    config.detector.template_threshold
                    if config.detector.mode == "template"
                    else config.detector.threshold
                )
                threshold_label = "templateThreshold" if config.detector.mode == "template" else "threshold"
                logging.info(
                    "Detector: label=%s confidence=%.3f %s=%.3f%s",
                    result.label,
                    result.confidence,
                    threshold_label,
                    threshold_value,
                    extra,
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
        override_file = crop_override_path(config_path)
        data = {
            "camera": {
                "crop": {
                    "enabled": bool(crop.enabled),
                    "x": int(crop.x),
                    "y": int(crop.y),
                    "w": int(crop.w),
                    "h": int(crop.h),
                }
            }
        }
        override_file.write_text(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        return


def _load_calibration_templates(template_dir: Path) -> List[Tuple[str, np.ndarray]]:
    templates: List[Tuple[str, np.ndarray]] = []
    if not template_dir.exists():
        return templates
    for item in template_dir.iterdir():
        if item.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        image = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        templates.append((item.name, image))
    return templates


def auto_calibrate_crop(camera: Camera, config: AppConfig, config_path: str) -> None:
    if not config.detector.enabled:
        return
    if config.detector.mode.lower() != "template":
        return
    if not config.detector.auto_calibrate:
        return

    template_dir = Path(config.detector.calibrate_template_dir)
    if not template_dir.exists():
        template_dir = Path(config.detector.template_dir)
    templates = _load_calibration_templates(template_dir)
    if not templates:
        logging.warning("Auto-calibrate skipped: no templates found in %s", template_dir)
        return

    try:
        frame = camera.read_raw()
    except Exception as exc:
        logging.warning("Auto-calibrate skipped: failed to read frame (%s)", exc)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    matches: List[Tuple[int, int, int, int, float, str]] = []
    best_score = 0.0

    for name, template in templates:
        best = 0.0
        best_loc = None
        best_size = None
        for scale in config.detector.template_scales:
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
            _, score, _, loc = cv2.minMaxLoc(result)
            score = float(score)
            if score > best:
                best = score
                best_loc = loc
                best_size = (scaled.shape[1], scaled.shape[0])
        if best_loc is None or best_size is None:
            continue
        if best >= config.detector.calibrate_min_score:
            x, y = best_loc
            w, h = best_size
            matches.append((x, y, w, h, best, name))
            if best > best_score:
                best_score = best

    effective_min_matches = max(2, config.detector.calibrate_min_matches)
    if len(matches) < effective_min_matches:
        logging.warning(
            "Auto-calibrate skipped: only %d template matches (min=%d).",
            len(matches),
            effective_min_matches,
        )
        return

    min_x = min(x for x, _, _, _, _, _ in matches)
    min_y = min(y for _, y, _, _, _, _ in matches)
    max_x = max(x + w for x, _, w, _, _, _ in matches)
    max_y = max(y + h for _, y, _, h, _, _ in matches)

    bbox_w = max(1, max_x - min_x)
    bbox_h = max(1, max_y - min_y)

    def compute_crop(use_frame_margins: bool) -> Tuple[int, int, int, int]:
        if use_frame_margins:
            left = int(frame_w * config.detector.calibrate_frame_margin_left)
            right = int(frame_w * config.detector.calibrate_frame_margin_right)
            top = int(frame_h * config.detector.calibrate_frame_margin_top)
            bottom = int(frame_h * config.detector.calibrate_frame_margin_bottom)
        else:
            left = int(bbox_w * config.detector.calibrate_margin_left)
            right = int(bbox_w * config.detector.calibrate_margin_right)
            top = int(bbox_h * config.detector.calibrate_margin_top)
            bottom = int(bbox_h * config.detector.calibrate_margin_bottom)
        crop_x = max(0, min_x - left)
        crop_y = max(0, min_y - top)
        crop_x2 = min(frame_w, max_x + right)
        crop_y2 = min(frame_h, max_y + bottom)
        crop_w = max(1, crop_x2 - crop_x)
        crop_h = max(1, crop_y2 - crop_y)
        return crop_x, crop_y, crop_w, crop_h

    crop_x, crop_y, crop_w, crop_h = compute_crop(config.detector.calibrate_use_frame_margins)
    if config.detector.calibrate_use_frame_margins:
        if crop_w >= int(frame_w * 0.98) or crop_h >= int(frame_h * 0.98):
            logging.warning(
                "Auto-calibrate: frame margins produced near-full-frame crop; falling back to bbox margins."
            )
            crop_x, crop_y, crop_w, crop_h = compute_crop(False)
        else:
            min_w = int(frame_w * 0.55)
            min_h = int(frame_h * 0.35)
            if crop_w < min_w or crop_h < min_h:
                logging.warning(
                    "Auto-calibrate: crop too small (w=%d h=%d); expanding to minimum frame fractions.",
                    crop_w,
                    crop_h,
                )
                target_w = max(crop_w, min_w)
                target_h = max(crop_h, min_h)
                expand_x = max(0, (target_w - crop_w) // 2)
                expand_y = max(0, (target_h - crop_h) // 2)
                crop_x = max(0, crop_x - expand_x)
                crop_y = max(0, crop_y - expand_y)
                crop_w = min(frame_w - crop_x, crop_w + expand_x * 2)
                crop_h = min(frame_h - crop_y, crop_h + expand_y * 2)

    config.camera.crop.enabled = True
    config.camera.crop.x = int(crop_x)
    config.camera.crop.y = int(crop_y)
    config.camera.crop.w = int(crop_w)
    config.camera.crop.h = int(crop_h)
    write_crop_to_config(config_path, config.camera.crop)
    logging.info(
        "Auto-calibrated crop: x=%d y=%d w=%d h=%d matches=%d best=%.3f",
        crop_x,
        crop_y,
        crop_w,
        crop_h,
        len(matches),
        best_score,
    )


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
        config_path = "config.yaml"
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
