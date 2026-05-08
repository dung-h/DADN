from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    CAMERA_INDEX,
    ESP32_CAMERA_PREVIEW_MODE,
    ESP32_CAMERA_URL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    INFER_EVERY_N_FRAMES,
    USE_ESP32_CAMERA,
)
from decision_engine import DecisionEngine
from detector import ObstacleDetector
from esp32_camera import ESP32CameraCapture
from model_utils import ensure_model
from tts_engine import VoiceAlertManager


def put_text_utf8(frame, text: str, xy, fontsize=20, color=(0, 255, 0)):
    """Draw UTF-8 text on frame using PIL."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    draw.text(xy, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_overlay(frame, obstacle, fps: float):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    if obstacle is None:
        frame = put_text_utf8(frame, "No priority obstacle", (10, 65), fontsize=18, color=(0, 255, 255))
        return frame

    cv2.rectangle(
        frame,
        (obstacle.x, obstacle.y),
        (obstacle.x + obstacle.w, obstacle.y + obstacle.h),
        (0, 255, 0),
        2,
    )
    text = f"{obstacle.label} | {obstacle.distance_level} | {obstacle.horizontal_zone}"
    if obstacle.ttc_seconds is not None:
        text += f" | TTC~{obstacle.ttc_seconds:.1f}s"
    frame = put_text_utf8(frame, text, (obstacle.x, max(20, obstacle.y - 10)), fontsize=16, color=(0, 255, 0))

    source_text = f"risk: {obstacle.risk_source}"
    frame = put_text_utf8(frame, source_text, (10, 65), fontsize=16, color=(120, 240, 240))
    frame = put_text_utf8(frame, obstacle.spoken_text, (10, FRAME_HEIGHT - 20), fontsize=16, color=(50, 220, 50))
    return frame


def main() -> None:
    ensure_model()

    detector = ObstacleDetector()
    decider = DecisionEngine(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    voice = VoiceAlertManager()

    if USE_ESP32_CAMERA:
        print(f"[INFO] Connecting to ESP32 camera at {ESP32_CAMERA_URL}...")
        cap = ESP32CameraCapture(ESP32_CAMERA_URL)
        camera_source = f"ESP32 ({ESP32_CAMERA_URL})"
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        camera_source = f"Local Camera (index {CAMERA_INDEX})"

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera: {camera_source}\n"
            f"Check config.py or the ESP32 camera IP."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print(f"✓ Camera source: {camera_source}")

    # Preview mode
    if ESP32_CAMERA_PREVIEW_MODE:
        print("[PREVIEW] Press 's' to start detection, 'q' to quit.")
        detection_active = False
        preview_frame_count = 0
        while not detection_active:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            preview_frame_count += 1
            
            if preview_frame_count % 10 == 0:
                frame = put_text_utf8(
                    frame, 
                    "[PREVIEW] Press 's' to start detection", 
                    (10, FRAME_HEIGHT - 20), 
                    fontsize=16, 
                    color=(0, 255, 255)
                )
            
            cv2.imshow("Obstacle Warning - PREVIEW MODE", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                detection_active = True
                cv2.destroyAllWindows()
                print("[DETECTION] Started. Press q to quit.")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    print("Press q to quit.")

    frame_count = 0
    start_time = time.time()
    current_obstacle = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Cannot read frame from camera.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1

        if frame_count % INFER_EVERY_N_FRAMES == 0:
            detections = detector.detect(frame)
            current_obstacle = decider.choose_alert(detections, timestamp=time.monotonic())
            if current_obstacle:
                ttc_text = (
                    f" | ttc={current_obstacle.ttc_seconds:.2f}s"
                    if current_obstacle.ttc_seconds is not None
                    else ""
                )
                print(
                    f"[ALERT] {current_obstacle.label} | {current_obstacle.distance_level} | "
                    f"{current_obstacle.horizontal_zone} | source={current_obstacle.risk_source}"
                    f"{ttc_text} | priority={current_obstacle.priority:.2f}"
                )
            voice.maybe_speak(current_obstacle)

        elapsed = max(time.time() - start_time, 1e-6)
        fps = frame_count / elapsed
        frame = draw_overlay(frame, current_obstacle, fps)

        cv2.imshow("Obstacle Warning", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
