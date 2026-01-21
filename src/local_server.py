from __future__ import annotations

import threading
from typing import Callable

from flask import Flask, Response, jsonify
import json


def start_local_server(
    port: int,
    capture_callback: Callable[[], None],
    preview_callback: Callable[[], bytes],
    preview_meta: dict,
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
      .frame { position: relative; display: inline-block; }
      img { max-width: 96vw; max-height: 90vh; border: 2px solid #23335a; display: block; }
      .crop-box {
        position: absolute;
        border: 2px dashed #2bdc74;
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.35);
        pointer-events: none;
      }
    </style>
  </head>
  <body>
    <header>ScoreboardCam preview (auto-refresh)</header>
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
      const cropBox = document.getElementById("cropBox");
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
      updateCrop();
      setInterval(() => {
        img.src = "/preview?t=" + Date.now();
      }, 1000);
    </script>
  </body>
</html>"""
        return Response(html, mimetype="text/html")

    thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": port}, daemon=True)
    thread.start()
    return thread
