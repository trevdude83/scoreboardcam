from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .config import DeviceConfig, ServerConfig


@dataclass
class ContextResponse:
    raw: Dict[str, object]

    @property
    def session_id(self) -> Optional[int]:
        active = self.raw.get("activeSession")
        if isinstance(active, dict):
            value = active.get("sessionId")
            return int(value) if isinstance(value, int) else None
        return None


class DeviceClient:
    def __init__(self, server: ServerConfig, device: DeviceConfig) -> None:
        self.base_url = server.base_url.rstrip("/")
        self.device = device

    def get_context(self) -> ContextResponse:
        url = f"{self.base_url}/api/v1/scoreboard/devices/{self.device.device_id}/context"
        response = self._request("GET", url)
        return ContextResponse(raw=response.json())

    def upload_images(
        self,
        images: List[bytes],
        session_id: Optional[int],
        captured_at: str,
        file_ext: str = "jpg",
        mime_type: str = "image/jpeg",
    ) -> Dict[str, object]:
        url = f"{self.base_url}/api/v1/scoreboard/ingest"
        files = []
        for index, image in enumerate(images):
            files.append(("images", (f"scoreboard-{index + 1}.{file_ext}", image, mime_type)))
        data: Dict[str, str] = {"capturedAt": captured_at, "deviceId": str(self.device.device_id)}
        if session_id:
            data["sessionId"] = str(session_id)
        response = self._request("POST", url, files=files, data=data)
        return response.json()

    def process_ingest(self, ingest_id: int) -> Dict[str, object]:
        url = f"{self.base_url}/api/v1/scoreboard/ingest/{ingest_id}/process"
        response = self._request("POST", url, json={"deviceId": self.device.device_id})
        return response.json()

    def _request(self, method: str, url: str, retries: int = 3, **kwargs: object) -> requests.Response:
        headers = kwargs.pop("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        headers.update(self._auth_header())
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                response = requests.request(method, url, headers=headers, timeout=10, **kwargs)
                if response.status_code >= 500:
                    raise RuntimeError(f"Server error {response.status_code}")
                if response.status_code >= 400:
                    raise RuntimeError(f"Request failed {response.status_code}: {response.text}")
                return response
            except Exception as exc:
                last_error = exc
                time.sleep(1.0 + attempt)
        raise RuntimeError(f"Request failed after retries: {last_error}")

    def _auth_header(self) -> Dict[str, str]:
        mode = self.device.auth_header_mode.lower()
        if mode == "bearer":
            return {"Authorization": f"Bearer {self.device.device_key}"}
        return {"X-Device-Key": self.device.device_key}
