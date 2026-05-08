from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite"
)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFER_EVERY_N_FRAMES = 2

# ========== ESP32 Camera Config ==========
# Set to True to use ESP32 camera instead of local webcam
USE_ESP32_CAMERA = True
# Thử các endpoint này nếu /stream không hoạt động:
# - http://10.121.219.227/ (homepage)
# - http://10.121.219.227/jpg (single frame - không stream)
# - http://10.121.219.227/mjpeg (motion jpeg stream)
# - http://10.121.219.227:81/stream (default port 81)
ESP32_CAMERA_URL = "http://10.121.219.227/"   # Or try /jpg, /mjpeg
ESP32_CAMERA_PREVIEW_MODE = False  # Set True to preview camera before detection starts (press 's' to start)
SCORE_THRESHOLD = 0.45
MAX_RESULTS = 8

# Fallback spatial risk thresholds used when TTC cannot be estimated yet.
NEAR_AREA_RATIO = 0.18
MEDIUM_AREA_RATIO = 0.06

# Center danger zone: objects here are more relevant to the user's path.
CENTER_ZONE_MIN_X = 0.25
CENTER_ZONE_MAX_X = 0.75
LOWER_ZONE_MIN_Y = 0.40

# TTC-proxy tuning.
# The system approximates collision risk from the short-term expansion
# of a tracked bounding box, not from metric depth.
TRACK_IOU_MIN = 0.35
TRACK_MAX_AGE_SECONDS = 1.20
TTC_MIN_SCALE_GROWTH = 0.03
TTC_IMMINENT_SECONDS = 0.90
TTC_SOON_SECONDS = 2.00
TTC_SMOOTHING_ALPHA = 0.65

SPEAK_COOLDOWN_SECONDS = 3.0
REPEAT_IF_RISK_UPGRADED_SECONDS = 1.0

# Only classes relevant to mobility safety should trigger alerts.
CLASS_WEIGHTS = {
    "person": 1.00,
    "bicycle": 0.90,
    "motorcycle": 1.00,
    "car": 0.95,
    "bus": 1.00,
    "truck": 1.00,
    "chair": 0.65,
    "bench": 0.60,
    "potted plant": 0.55,
    "suitcase": 0.60,
    "backpack": 0.45,
}

VI_LABELS = {
    "person": "người",
    "bicycle": "xe đạp",
    "motorcycle": "xe máy",
    "car": "ô tô",
    "bus": "xe buýt",
    "truck": "xe tải",
    "chair": "ghế",
    "bench": "ghế dài",
    "potted plant": "chậu cây",
    "suitcase": "va li",
    "backpack": "ba lô",
}

EN_LABELS = {
    "person": "person",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "chair": "chair",
    "bench": "bench",
    "potted plant": "plant",
    "suitcase": "suitcase",
    "backpack": "backpack",
}
