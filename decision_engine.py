from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from config import (
    CENTER_ZONE_MAX_X,
    CENTER_ZONE_MIN_X,
    CLASS_WEIGHTS,
    EN_LABELS,
    LOWER_ZONE_MIN_Y,
    MEDIUM_AREA_RATIO,
    NEAR_AREA_RATIO,
    TRACK_IOU_MIN,
    TRACK_MAX_AGE_SECONDS,
    TTC_IMMINENT_SECONDS,
    TTC_MIN_SCALE_GROWTH,
    TTC_SMOOTHING_ALPHA,
    TTC_SOON_SECONDS,
    VI_LABELS,
)
from detector import DetectionItem


@dataclass
class TrackState:
    label: str
    x: int
    y: int
    w: int
    h: int
    last_seen_at: float
    scale_measure: float
    smoothed_ttc_seconds: Optional[float] = None


@dataclass
class ScoredObstacle:
    label: str
    label_vi: str
    score: float
    x: int
    y: int
    w: int
    h: int
    area_ratio: float
    horizontal_zone: str
    distance_level: str
    is_in_center_zone: bool
    priority: float
    spoken_text: str
    risk_source: str
    ttc_seconds: Optional[float]
    scale_growth: float


class DecisionEngine:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        self._tracks: dict[int, TrackState] = {}
        self._next_track_id = 1

    def choose_alert(
        self,
        detections: Iterable[DetectionItem],
        timestamp: Optional[float] = None,
    ) -> Optional[ScoredObstacle]:
        now = time.monotonic() if timestamp is None else timestamp
        valid_detections = [item for item in detections if item.label in CLASS_WEIGHTS]
        self._prune_tracks(now)
        matches = self._match_tracks(valid_detections)

        candidates = []
        next_tracks: dict[int, TrackState] = {}

        for index, det in enumerate(valid_detections):
            track_id = matches.get(index)
            previous = self._tracks.get(track_id) if track_id is not None else None
            obstacle, track_state = self._score_detection(det, previous, now)
            if obstacle is not None:
                candidates.append(obstacle)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
            next_tracks[track_id] = track_state

        self._tracks = next_tracks

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.priority)

    def _score_detection(
        self,
        det: DetectionItem,
        previous: Optional[TrackState],
        now: float,
    ) -> tuple[Optional[ScoredObstacle], TrackState]:
        class_weight = CLASS_WEIGHTS.get(det.label)
        if class_weight is None:
            track_state = self._build_track_state(det, now, None)
            return None, track_state

        area_ratio = (det.w * det.h) / max(self.frame_area, 1)
        height_ratio = det.h / max(self.frame_height, 1)
        center_x_ratio = (det.x + det.w / 2) / self.frame_width
        bottom_y_ratio = (det.y + det.h) / self.frame_height

        horizontal_zone = self._horizontal_zone(center_x_ratio)
        is_in_center_zone = CENTER_ZONE_MIN_X <= center_x_ratio <= CENTER_ZONE_MAX_X
        is_in_lower_zone = bottom_y_ratio >= LOWER_ZONE_MIN_Y
        scale_measure = self._scale_measure(area_ratio, height_ratio)

        ttc_seconds = None
        scale_growth = 0.0
        smoothed_ttc = None
        if previous is not None:
            dt = max(now - previous.last_seen_at, 1e-6)
            if dt <= TRACK_MAX_AGE_SECONDS:
                scale_growth = (scale_measure - previous.scale_measure) / max(previous.scale_measure, 1e-6)
                if scale_growth >= TTC_MIN_SCALE_GROWTH:
                    ttc_seconds = dt / scale_growth
                    smoothed_ttc = self._smooth_ttc(previous.smoothed_ttc_seconds, ttc_seconds)

        spatial_level, spatial_weight = self._spatial_level(area_ratio, is_in_lower_zone)
        risk_level, risk_weight, risk_source = self._risk_from_ttc(smoothed_ttc, spatial_level, spatial_weight)

        center_weight = 1.18 if is_in_center_zone else 0.86
        direction_weight = 1.00 if horizontal_zone == "center" else 0.92
        confidence_weight = max(0.4, det.score)

        priority = (
            class_weight
            * risk_weight
            * center_weight
            * direction_weight
            * confidence_weight
        )

        label_vi = VI_LABELS.get(det.label, det.label)
        label_en = EN_LABELS.get(det.label, det.label.replace("_", " "))
        spoken_text = self._spoken_text(label_en, risk_level, horizontal_zone, smoothed_ttc, risk_source)
        track_state = self._build_track_state(det, now, smoothed_ttc, scale_measure)

        return (
            ScoredObstacle(
                label=det.label,
                label_vi=label_vi,
                score=det.score,
                x=det.x,
                y=det.y,
                w=det.w,
                h=det.h,
                area_ratio=area_ratio,
                horizontal_zone=horizontal_zone,
                distance_level=risk_level,
                is_in_center_zone=is_in_center_zone,
                priority=priority,
                spoken_text=spoken_text,
                risk_source=risk_source,
                ttc_seconds=smoothed_ttc,
                scale_growth=scale_growth,
            ),
            track_state,
        )

    def _build_track_state(
        self,
        det: DetectionItem,
        now: float,
        smoothed_ttc_seconds: Optional[float],
        scale_measure: Optional[float] = None,
    ) -> TrackState:
        area_ratio = (det.w * det.h) / max(self.frame_area, 1)
        if scale_measure is None:
            scale_measure = self._scale_measure(area_ratio, det.h / max(self.frame_height, 1))
        return TrackState(
            label=det.label,
            x=det.x,
            y=det.y,
            w=det.w,
            h=det.h,
            last_seen_at=now,
            scale_measure=scale_measure,
            smoothed_ttc_seconds=smoothed_ttc_seconds,
        )

    def _prune_tracks(self, now: float) -> None:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if (now - track.last_seen_at) <= TRACK_MAX_AGE_SECONDS
        }

    def _match_tracks(self, detections: list[DetectionItem]) -> dict[int, int]:
        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        candidate_pairs: list[tuple[float, int, int]] = []

        for det_index, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.label != det.label:
                    continue
                iou = self._iou(det, track)
                if iou >= TRACK_IOU_MIN:
                    candidate_pairs.append((iou, det_index, track_id))

        candidate_pairs.sort(reverse=True)
        for _, det_index, track_id in candidate_pairs:
            if det_index in matches or track_id in used_tracks:
                continue
            matches[det_index] = track_id
            used_tracks.add(track_id)
        return matches

    @staticmethod
    def _scale_measure(area_ratio: float, height_ratio: float) -> float:
        return 0.7 * height_ratio + 0.3 * math.sqrt(max(area_ratio, 0.0))

    @staticmethod
    def _smooth_ttc(previous_ttc: Optional[float], current_ttc: float) -> float:
        if previous_ttc is None:
            return current_ttc
        return (TTC_SMOOTHING_ALPHA * previous_ttc) + ((1.0 - TTC_SMOOTHING_ALPHA) * current_ttc)

    @staticmethod
    def _horizontal_zone(center_x_ratio: float) -> str:
        if center_x_ratio < 0.33:
            return "left"
        if center_x_ratio > 0.67:
            return "right"
        return "center"

    @staticmethod
    def _spatial_level(area_ratio: float, is_in_lower_zone: bool) -> tuple[str, float]:
        adjusted_ratio = area_ratio * (1.1 if is_in_lower_zone else 1.0)
        if adjusted_ratio >= NEAR_AREA_RATIO:
            return "near", 1.20
        if adjusted_ratio >= MEDIUM_AREA_RATIO:
            return "medium", 1.00
        return "far", 0.80

    @staticmethod
    def _risk_from_ttc(
        ttc_seconds: Optional[float],
        spatial_level: str,
        spatial_weight: float,
    ) -> tuple[str, float, str]:
        if ttc_seconds is None:
            return spatial_level, spatial_weight, "spatial_fallback"
        if ttc_seconds <= TTC_IMMINENT_SECONDS:
            return "near", 1.40, "ttc_proxy"
        if ttc_seconds <= TTC_SOON_SECONDS:
            return "medium", 1.12, "ttc_proxy"
        return "far", 0.85, "ttc_proxy"

    @staticmethod
    def _spoken_text(
        label_en: str,
        risk_level: str,
        horizontal_zone: str,
        ttc_seconds: Optional[float],
        risk_source: str,
    ) -> str:
        zone_phrase = "ahead" if horizontal_zone == "center" else f"on the {horizontal_zone}"
        if risk_source == "ttc_proxy" and ttc_seconds is not None:
            if risk_level == "near":
                return f"Warning, {label_en} approaching fast {zone_phrase}"
            if risk_level == "medium":
                return f"{label_en} approaching {zone_phrase}"
        if risk_level == "near":
            return f"Warning, {label_en} very close {zone_phrase}"
        if risk_level == "medium":
            return f"{label_en} ahead at medium risk {zone_phrase}"
        return f"{label_en} {zone_phrase}"

    @staticmethod
    def _iou(det: DetectionItem, track: TrackState) -> float:
        det_x2 = det.x + det.w
        det_y2 = det.y + det.h
        track_x2 = track.x + track.w
        track_y2 = track.y + track.h

        inter_x1 = max(det.x, track.x)
        inter_y1 = max(det.y, track.y)
        inter_x2 = min(det_x2, track_x2)
        inter_y2 = min(det_y2, track_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h
        if intersection <= 0:
            return 0.0

        det_area = det.w * det.h
        track_area = track.w * track.h
        union = det_area + track_area - intersection
        return intersection / max(union, 1)
