from __future__ import annotations

import threading
from typing import Callable

from flask import Flask, jsonify


def start_local_server(port: int, capture_callback: Callable[[], None]) -> threading.Thread:
    app = Flask("scoreboardcam")

    @app.route("/capture", methods=["POST", "GET"])
    def capture() -> tuple:
        capture_callback()
        return jsonify({"ok": True})

    thread = threading.Thread(target=app.run, kwargs={"host": "0.0.0.0", "port": port}, daemon=True)
    thread.start()
    return thread
