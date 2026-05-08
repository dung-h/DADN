"""
Test different ESP32 camera endpoints to find the correct stream URL.
Run: python test_esp32_endpoints.py
"""

import cv2
import requests
import sys
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Test ESP32 camera endpoints.")
parser.add_argument("ip", nargs="?", default="10.121.219.227", help="ESP32 camera IP address")
args = parser.parse_args()

# Test endpoints (try them in order)
ESP32_IP = args.ip
TEST_ENDPOINTS = [
    f"http://{ESP32_IP}/",
    f"http://{ESP32_IP}/jpg",
    f"http://{ESP32_IP}:80/stream",
    f"http://{ESP32_IP}:81/stream",
    f"http://{ESP32_IP}:81/",
    f"http://{ESP32_IP}/mjpeg",
    f"http://{ESP32_IP}:8080/stream",
    f"http://{ESP32_IP}/cam-mid.jpg",
    f"http://{ESP32_IP}/cam.jpg",
]

print("=" * 60)
print("🔍 ESP32 Camera Endpoint Tester")
print("=" * 60)

# Test 1: Try MJPEG stream endpoints
print("\n[PHASE 1] Testing MJPEG Stream Endpoints...")
for endpoint in TEST_ENDPOINTS:
    print(f"\n➤ Testing: {endpoint}")
    
    try:
        cap = cv2.VideoCapture(endpoint)
        
        # Try to read a frame with timeout
        success = False
        for _ in range(5):  # Try up to 5 times
            ret, frame = cap.read()
            if ret:
                print(f"   ✅ SUCCESS! Got frame: {frame.shape}")
                success = True
                cap.release()
                
                print(f"\n{'='*60}")
                print(f"✓ Found working endpoint: {endpoint}")
                print(f"{'='*60}")
                print(f"\nUpdate your config.py:")
                print(f"  USE_ESP32_CAMERA = True")
                print(f"  ESP32_CAMERA_URL = \"{endpoint}\"")
                print(f"\nThen run: python main.py  (or python web_ui.py for web UI)")
                sys.exit(0)
        
        if not success:
            print(f"   ❌ No valid frames received")
        cap.release()
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:60]}")

# Test 2: Try HTTP GET with polling (fallback for /jpg endpoint)
print(f"\n{'='*60}")
print("\n[PHASE 2] Testing JPG Polling (HTTP GET)...")

jpg_endpoints = [
    f"http://{ESP32_IP}/jpg",
    f"http://{ESP32_IP}/cam.jpg",
    f"http://{ESP32_IP}/cam-mid.jpg",
    f"http://{ESP32_IP}/capture",
]

for endpoint in jpg_endpoints:
    print(f"\n➤ Testing JPG endpoint: {endpoint}")
    try:
        resp = requests.get(endpoint, timeout=3)
        if resp.status_code == 200 and resp.headers.get('content-type', '').startswith('image'):
            print(f"   ✅ Got image! ({len(resp.content)} bytes)")
            print(f"\n{'='*60}")
            print(f"✓ Found JPG endpoint: {endpoint}")
            print(f"{'='*60}")
            print(f"\nUse this in config.py:")
            print(f"  USE_ESP32_CAMERA = True")
            print(f"  ESP32_CAMERA_URL = \"http_jpg://{endpoint}\"")
            print(f"\nThis will poll the JPG endpoint for frames.")
            break
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:60]}")

print(f"\n{'='*60}")
print("❌ None of the endpoints worked!")
print("=" * 60)
print("\nTroubleshooting:")
print("1. Check if ESP32 is powered on and accessible")
print(f"2. Verify IP address (try browsing http://{ESP32_IP}/ in browser)")
print("3. Check firewall settings")
print("4. Try different ports (e.g., :80, :81, :8080)")
print("5. Check your ESP32 firmware docs:")
print("   - Ai-thinker firmware: typically /jpg or /stream")
print("   - Arduino esp32 examples: check the specific sketch")
print("   - MotionEye or other: check project documentation")

