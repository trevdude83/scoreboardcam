from __future__ import annotations

import threading
from typing import Callable

import logging
from flask import Flask, Response, jsonify, request
import json


def start_local_server(
    port: int,
    capture_callback: Callable[[], None],
    preview_callback: Callable[[], bytes],
    preview_meta: dict,
    update_crop_callback: Callable[[dict], dict],
    probe_callback: Callable[[int, float], dict] | None = None,
) -> threading.Thread:
    app = Flask("scoreboardcam")

    @app.route("/capture", methods=["POST", "GET"])
    def capture() -> tuple:
        capture_callback()
        return jsonify({"ok": True})

    @app.route("/preview", methods=["GET"])
    def preview() -> Response:
        image_bytes = preview_callback()
        return Response(image_bytes, mimetype="image/jpeg")

    @app.route("/preview.html", methods=["GET"])
    def preview_page() -> Response:
        meta_json = json.dumps(preview_meta)
        html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ScoreboardCam Preview</title>
    <style>
      body { margin: 0; font-family: Arial, sans-serif; background: #0b0f1a; color: #fff; }
      header { padding: 12px 16px; background: #121a2e; font-size: 14px; }
      main { display: flex; justify-content: center; align-items: center; padding: 16px; }
      .frame { position: relative; display: inline-block; touch-action: none; }
      img { max-width: 96vw; max-height: 90vh; border: 2px solid #23335a; display: block; user-select: none; -webkit-user-drag: none; }
      .crop-box {
        position: absolute;
        border: 2px dashed #2bdc74;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.35);
        pointer-events: none;
      }
      .controls {
        padding: 12px 16px;
        display: grid;
        grid-template-columns: repeat(6, minmax(90px, 1fr));
        gap: 8px;
        background: #0f1628;
      }
      .controls label { font-size: 12px; display: grid; gap: 4px; }
      .controls input { padding: 6px; border-radius: 6px; border: 1px solid #2b3c6a; background: #0b1224; color: #fff; }
      .controls button { grid-column: span 2; padding: 8px 12px; background: #2d6bff; border: none; color: #fff; border-radius: 6px; cursor: pointer; }
      .controls .toggle { display: flex; align-items: center; gap: 6px; }
      .frame.dragging { cursor: crosshair; }
    </style>
  </head>
  <body>
    <header>ScoreboardCam preview (auto-refresh)</header>
    <section class="controls">
      <label class="toggle">
        <input id="enabled" type="checkbox" />
        Crop enabled
      </label>
      <label>X<input id="x" type="number" /></label>
      <label>Y<input id="y" type="number" /></label>
      <label>W<input id="w" type="number" /></label>
      <label>H<input id="h" type="number" /></label>
      <button id="save">Save crop</button>
    </section>
    <main>
      <div class="frame">
        <img id="preview" src="/preview" alt="Preview" />
        <div id="cropBox" class="crop-box"></div>
      </div>
    </main>
    <script>
      const meta = """ + meta_json + """;
      const crop = meta.crop || { enabled: false };
      const img = document.getElementById("preview");
      img.draggable = false;
      const cropBox = document.getElementById("cropBox");
      const enabled = document.getElementById("enabled");
      const inputX = document.getElementById("x");
      const inputY = document.getElementById("y");
      const inputW = document.getElementById("w");
      const inputH = document.getElementById("h");
      const save = document.getElementById("save");
      const frame = document.querySelector(".frame");
      let dragStart = null;
      let imageReady = false;
      function fillInputs() {
        enabled.checked = !!crop.enabled;
        inputX.value = String(crop.x ?? 0);
        inputY.value = String(crop.y ?? 0);
        inputW.value = String(crop.w ?? meta.width ?? 0);
        inputH.value = String(crop.h ?? meta.height ?? 0);
      }
      function updateCrop() {
        if (!crop.enabled || !meta.width || !meta.height) {
          cropBox.style.display = "none";
          return;
        }
        cropBox.style.display = "block";
        cropBox.style.left = (crop.x / meta.width * 100) + "%";
        cropBox.style.top = (crop.y / meta.height * 100) + "%";
        cropBox.style.width = (crop.w / meta.width * 100) + "%";
        cropBox.style.height = (crop.h / meta.height * 100) + "%";
      }
      function applyDrag(x1, y1, x2, y2) {
        if (!imageReady) return;
        const imgRect = img.getBoundingClientRect();
        if (!imgRect.width || !imgRect.height) return;
        const left = Math.max(0, Math.min(x1, x2) - imgRect.left);
        const top = Math.max(0, Math.min(y1, y2) - imgRect.top);
        const right = Math.min(imgRect.width, Math.max(x1, x2) - imgRect.left);
        const bottom = Math.min(imgRect.height, Math.max(y1, y2) - imgRect.top);
        const scaleX = meta.width / imgRect.width;
        const scaleY = meta.height / imgRect.height;
        crop.enabled = true;
        crop.x = Math.round(left * scaleX);
        crop.y = Math.round(top * scaleY);
        crop.w = Math.max(1, Math.round((right - left) * scaleX));
        crop.h = Math.max(1, Math.round((bottom - top) * scaleY));
        fillInputs();
        updateCrop();
      }
      fillInputs();
      updateCrop();
      img.addEventListener("load", () => {
        imageReady = true;
        updateCrop();
      });
      setInterval(() => {
        img.src = "/preview?t=" + Date.now();
      }, 1000);
      function beginDrag(event) {
        if (!meta.width || !meta.height) return;
        if (!dragStart) {
          dragStart = { x: event.clientX, y: event.clientY };
          frame.classList.add("dragging");
          applyDrag(dragStart.x, dragStart.y, event.clientX, event.clientY);
          if (event.pointerId !== undefined) {
            frame.setPointerCapture(event.pointerId);
          }
          return;
        }
        applyDrag(dragStart.x, dragStart.y, event.clientX, event.clientY);
        dragStart = null;
        frame.classList.remove("dragging");
        if (event.pointerId !== undefined) {
          frame.releasePointerCapture(event.pointerId);
        }
      }
      function moveDrag(event) {
        if (!dragStart) return;
        applyDrag(dragStart.x, dragStart.y, event.clientX, event.clientY);
      }
      frame.addEventListener("pointerdown", beginDrag);
      frame.addEventListener("pointermove", moveDrag);
      save.addEventListener("click", async () => {
        const payload = {
          enabled: enabled.checked,
          x: Number(inputX.value),
          y: Number(inputY.value),
          w: Number(inputW.value),
          h: Number(inputH.value),
        };
        const res = await fetch("/crop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          const data = await res.json();
          meta.crop = data.crop;
          Object.assign(crop, data.crop);
          fillInputs();
          updateCrop();
        }
      });
    </script>
  </body>
</html>"""
        return Response(html, mimetype="text/html")

    @app.route("/probe", methods=["GET"])
    def probe() -> Response:
        if probe_callback is None:
            return jsonify({"error": "Detector probe not available."}), 400
        try:
            count = int(request.args.get("count", 20))
        except (TypeError, ValueError):
            count = 20
        try:
            delay_ms = float(request.args.get("delayMs", 50))
        except (TypeError, ValueError):
            delay_ms = 50.0
        count = max(1, min(count, 200))
        delay_ms = max(0.0, min(delay_ms, 2000.0))
        payload = probe_callback(count, delay_ms / 1000.0)
        logging.info("Detector probe: %s", payload)
        return jsonify(payload)

    @app.route("/crop", methods=["POST"])
    def update_crop() -> Response:
        data = request.get_json(silent=True) or {}
        updated = update_crop_callback(data)
        preview_meta["crop"] = updated
        return jsonify({"crop": updated})

    thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": port}, daemon=True)
    thread.start()
    return thread
