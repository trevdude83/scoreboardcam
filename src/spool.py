from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SpoolItem:
    image_path: Path
    meta_path: Path
    metadata: Dict[str, object]


def save_to_spool(spool_dir: Path, image_bytes: bytes, metadata: Dict[str, object]) -> SpoolItem:
    spool_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time() * 1000)
    extension = str(metadata.get("format") or "jpg")
    image_path = spool_dir / f"scoreboard-{stamp}.{extension}"
    meta_path = spool_dir / f"scoreboard-{stamp}.json"
    image_path.write_bytes(image_bytes)
    meta_path.write_text(json.dumps(metadata, indent=2))
    return SpoolItem(image_path=image_path, meta_path=meta_path, metadata=metadata)


def list_spool(spool_dir: Path) -> List[SpoolItem]:
    items: List[SpoolItem] = []
    if not spool_dir.exists():
        return items
    for meta_path in sorted(spool_dir.glob("*.json")):
        try:
            metadata = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            metadata = {}
        extension = str(metadata.get("format") or "jpg")
        image_name = meta_path.name.replace(".json", f".{extension}")
        image_path = spool_dir / image_name
        if image_path.exists():
            items.append(SpoolItem(image_path=image_path, meta_path=meta_path, metadata=metadata))
    return items


def delete_spool_item(item: SpoolItem) -> None:
    if item.image_path.exists():
        item.image_path.unlink()
    if item.meta_path.exists():
        item.meta_path.unlink()
