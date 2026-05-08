"""
Flask web UI for live stream and detection control.
Run: python web_ui.py
Access: http://localhost:5000
"""

from __future__ import annotations

import cv2
import numpy as np
from flask import Flask, render_template_string, Response
from threading import Lock
import time

from config import (
    CAMERA_INDEX,
    ESP32_CAMERA_URL,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    USE_ESP32_CAMERA,
)
from detector import ObstacleDetector
from decision_engine import DecisionEngine
from esp32_camera import ESP32CameraCapture
from model_utils import ensure_model
from tts_engine import VoiceAlertManager

app = Flask(__name__)

# Shared state
camera_lock = Lock()
cap = None
detector = None
decider = None
voice = None
detection_enabled = True
last_frame_time = 0
current_obstacle = None
frame_counter = 0


def init_camera():
    global cap
    if USE_ESP32_CAMERA:
        print(f"[INFO] Connecting to ESP32 camera at {ESP32_CAMERA_URL}...")
        cap = ESP32CameraCapture(ESP32_CAMERA_URL)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    print("[OK] Camera initialized")


def init_detector():
    global detector, decider, voice
    ensure_model()
    detector = ObstacleDetector()
    decider = DecisionEngine(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    voice = VoiceAlertManager()
    print("[OK] Detector initialized")


def _read_processed_frame():
    """Read, annotate, and return a single frame for web rendering."""
    global current_obstacle, last_frame_time, frame_counter, cap

    if cap is None:
        return None

    with camera_lock:
        ok, frame = cap.read()
        if not ok and not USE_ESP32_CAMERA:
            # Local webcams on Windows can occasionally stall in a long-lived process.
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_counter += 1

        if detection_enabled and frame_counter % 2 == 0:
            detections = detector.detect(frame)
            current_obstacle = decider.choose_alert(detections)
            if current_obstacle:
                voice.maybe_speak(current_obstacle)

        if current_obstacle and detection_enabled:
            cv2.rectangle(
                frame,
                (current_obstacle.x, current_obstacle.y),
                (current_obstacle.x + current_obstacle.w, current_obstacle.y + current_obstacle.h),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{current_obstacle.label} | {current_obstacle.distance_level}",
                (current_obstacle.x, max(20, current_obstacle.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        elapsed = max(time.time() - last_frame_time, 1e-6)
        fps = 1.0 / elapsed if elapsed > 0 else 0
        last_frame_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | {'DETECTION ON' if detection_enabled else 'DETECTION OFF'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if detection_enabled else (100, 100, 100),
            2,
        )
        return frame


def gen_frames():
    """Generate MJPEG frames directly from the active camera."""
    while True:
        frame = _read_processed_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            time.sleep(0.05)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )
        time.sleep(0.01)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DADN - Obstacle Warning System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #1e1e1e;
            color: #fff;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: #2d2d2d;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            max-width: 800px;
            width: 100%;
        }
        h1 {
            text-align: center;
            margin: 0 0 20px 0;
            color: #00ff00;
        }
        .video-container {
            text-align: center;
            margin-bottom: 20px;
            background: #000;
            border-radius: 5px;
            overflow: hidden;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-bottom: 20px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            background: #00aa00;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #00dd00;
        }
        button.off {
            background: #aa0000;
        }
        button.off:hover {
            background: #dd0000;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: #3d3d3d;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .status p {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚧 DADN - Obstacle Warning System</h1>
        
        <div class="video-container">
            <img id="cameraFeed" src="/snapshot.jpg?ts=0" alt="Camera Feed" style="width:100%;">
        </div>
        
        <div class="controls">
            <button id="detectionBtn" onclick="toggleDetection()">
                START DETECTION
            </button>
            <button onclick="location.reload()">REFRESH</button>
        </div>
        
        <div class="status">
            <p id="status">Detection Status: <strong>LOADING...</strong></p>
            <p style="font-size: 12px; margin-top: 10px; opacity: 0.7;">
                Press 'q' in the OpenCV window to quit (if running locally)
            </p>
        </div>
    </div>
    
    <script>
        let detectionActive = true;
        
        function toggleDetection() {
            const btn = document.getElementById('detectionBtn');
            
            fetch('/toggle_detection')
                .then(r => r.json())
                .then(data => {
                    detectionActive = data.detection_enabled;
                    updateUI();
                })
                .catch(err => console.error('Error:', err));
        }
        
        function updateUI() {
            const btn = document.getElementById('detectionBtn');
            const status = document.getElementById('status');
            
            if (detectionActive) {
                btn.textContent = '🛑 STOP DETECTION';
                btn.classList.remove('off');
                status.innerHTML = 'Detection Status: <strong style="color: #00ff00;">ON</strong>';
            } else {
                btn.textContent = '▶️  START DETECTION';
                btn.classList.add('off');
                status.innerHTML = 'Detection Status: <strong style="color: #ff0000;">OFF</strong>';
            }
        }

        function refreshFrame() {
            const img = document.getElementById('cameraFeed');
            img.src = '/snapshot.jpg?ts=' + Date.now();
        }
        
        // Initial state
        updateUI();
        setInterval(refreshFrame, 250);
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/snapshot.jpg')
def snapshot():
    frame = _read_processed_frame()
    if frame is None:
        return Response(status=503)

    ok, buffer = cv2.imencode('.jpg', frame)
    if not ok:
        return Response(status=500)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/toggle_detection')
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return {'detection_enabled': detection_enabled}


def main():
    global cap
    
    try:
        init_camera()
        init_detector()

        print("\n" + "="*50)
        print("[WEB] UI started: http://localhost:5000")
        print("="*50 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if cap:
            cap.release()


if __name__ == '__main__':
    main()
