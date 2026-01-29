from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from .detector import DetectionResult


@dataclass
class DebounceDecision:
    triggered: bool
    best_index: Optional[int]


class DebouncedWindow:
    def __init__(
        self,
        scoreboard_label: str,
        threshold: float,
        required_hits: int,
        window_size: int,
        cooldown_seconds: int,
        rearm_min_clears: int,
    ) -> None:
        self.scoreboard_label = scoreboard_label
        self.threshold = threshold
        self.required_hits = required_hits
        self.window_size = window_size
        self.cooldown_seconds = cooldown_seconds
        self.rearm_min_clears = max(1, rearm_min_clears)
        self.window: Deque[DetectionResult] = deque(maxlen=window_size)
        self.last_trigger_time = 0.0
        self.armed = True

    def update(self, result: DetectionResult) -> DebounceDecision:
        self.window.append(result)
        now = time.time()
        if not self.armed:
            if len(self.window) >= self.rearm_min_clears:
                tail = list(self.window)[-self.rearm_min_clears :]
                cleared = all(
                    entry.label != self.scoreboard_label or entry.confidence < self.threshold
                    for entry in tail
                )
                if cleared:
                    self.armed = True
            return DebounceDecision(triggered=False, best_index=None)
        if now - self.last_trigger_time < self.cooldown_seconds:
            return DebounceDecision(triggered=False, best_index=None)

        hits = [
            idx
            for idx, entry in enumerate(self.window)
            if entry.label == self.scoreboard_label and entry.confidence >= self.threshold
        ]
        if len(hits) < self.required_hits:
            return DebounceDecision(triggered=False, best_index=None)

        best_index = max(hits, key=lambda idx: self.window[idx].confidence)
        self.last_trigger_time = now
        self.armed = False
        return DebounceDecision(triggered=True, best_index=best_index)
