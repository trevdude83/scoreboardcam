from __future__ import annotations

import threading
from typing import Callable

from flask import Flask, Response, jsonify


def start_local_server(port: int, capture_callback: Callable[[], None], preview_callback: Callable[[], bytes]) -> threading.Thread:
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
      img { max-width: 96vw; max-height: 90vh; border: 2px solid #23335a; }
    </style>
  </head>
  <body>
    <header>ScoreboardCam preview (auto-refresh)</header>
    <main>
      <img id="preview" src="/preview" alt="Preview" />
    </main>
    <script>
      const img = document.getElementById("preview");
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
