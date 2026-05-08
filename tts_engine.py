from __future__ import annotations

import time
from typing import Optional

import pyttsx3

from config import REPEAT_IF_RISK_UPGRADED_SECONDS, SPEAK_COOLDOWN_SECONDS
from decision_engine import ScoredObstacle


_DISTANCE_RANK = {"far": 0, "medium": 1, "near": 2}


class VoiceAlertManager:
    def __init__(self) -> None:
        self.tts = pyttsx3.init()
        self._prefer_english_voice()
        self.tts.setProperty("rate", 165)
        self.last_spoken_at = 0.0
        self.last_message: Optional[str] = None
        self.last_distance_level: Optional[str] = None

    def _prefer_english_voice(self) -> None:
        voices = self.tts.getProperty("voices") or []
        for voice in voices:
            voice_blob = " ".join(
                str(part).lower()
                for part in (getattr(voice, "id", ""), getattr(voice, "name", ""), getattr(voice, "languages", ""))
            )
            if any(token in voice_blob for token in ("en", "english", "us", "uk")):
                self.tts.setProperty("voice", voice.id)
                return

    def maybe_speak(self, obstacle: Optional[ScoredObstacle]) -> bool:
        if obstacle is None:
            return False

        now = time.time()
        current_rank = _DISTANCE_RANK.get(obstacle.distance_level, 0)
        last_rank = _DISTANCE_RANK.get(self.last_distance_level or "far", 0)
        risk_upgraded = current_rank > last_rank and (now - self.last_spoken_at) >= REPEAT_IF_RISK_UPGRADED_SECONDS
        cooldown_passed = (now - self.last_spoken_at) >= SPEAK_COOLDOWN_SECONDS
        new_message = obstacle.spoken_text != self.last_message

        if cooldown_passed or (risk_upgraded and new_message):
            self.tts.say(obstacle.spoken_text)
            self.tts.runAndWait()
            self.last_spoken_at = now
            self.last_message = obstacle.spoken_text
            self.last_distance_level = obstacle.distance_level
            return True
        return False
