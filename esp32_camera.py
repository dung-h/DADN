"""
Camera wrapper to support both MJPEG stream and HTTP JPG polling.
"""

import cv2
import requests
import numpy as np
from io import BytesIO
import time
from threading import Thread, Lock, Event

class ESP32CameraCapture:
    """
    Wrapper for ESP32 camera that supports:
    - MJPEG stream (opencv VideoCapture)
    - HTTP JPG polling (fallback for /jpg endpoint)
    """
    
    def __init__(self, url: str):
        self.url = url
        self._mode = None  # Will be set on first read
        self._cap = None
        self._jpg_url = None
        self._jpg_thread = None
        self._jpg_frame = None
        self._jpg_lock = Lock()
        self._stop_event = Event()
        self._fps = 0
        self._last_time = time.time()
        
    def _jpg_polling_thread(self):
        """Thread that continuously polls JPG endpoint."""
        while not self._stop_event.is_set():
            try:
                resp = requests.get(self._jpg_url, timeout=5)
                if resp.status_code == 200:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._jpg_lock:
                            self._jpg_frame = frame
            except Exception as e:
                print(f"[JPG Polling Error] {e}")
            time.sleep(0.05)  # ~20 FPS polling
    
    def isOpened(self):
        """Check if camera is opened."""
        if self._mode is None:
            ok, _ = self.read()
            return ok
        if self._mode == 'stream':
            return self._cap is not None and self._cap.isOpened()
        elif self._mode == 'jpg':
            return self._jpg_frame is not None
        return False
    
    def read(self):
        """Read frame - will auto-detect mode on first call."""
        
        # First read - detect mode
        if self._mode is None:
            print(f"[ESP32 Camera] Detecting mode for: {self.url}")
            
            # Try MJPEG stream first
            try:
                print("  → Trying MJPEG stream mode...")
                self._cap = cv2.VideoCapture(self.url)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    self._mode = 'stream'
                    print("  ✓ Using MJPEG stream mode")
                    return ret, frame
                self._cap.release()
                self._cap = None
            except Exception as e:
                print(f"  ✗ MJPEG stream failed: {str(e)[:50]}")
            
            # Try HTTP JPG polling as fallback
            try:
                print("  → Trying HTTP JPG polling mode...")
                self._jpg_url = self.url
                resp = requests.get(self._jpg_url, timeout=5)
                if resp.status_code == 200:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._mode = 'jpg'
                        with self._jpg_lock:
                            self._jpg_frame = frame
                        # Start polling thread
                        self._stop_event.clear()
                        self._jpg_thread = Thread(target=self._jpg_polling_thread, daemon=True)
                        self._jpg_thread.start()
                        print("  ✓ Using HTTP JPG polling mode")
                        return True, frame
            except Exception as e:
                print(f"  ✗ HTTP JPG polling failed: {str(e)[:50]}")
            
            print("  ✗ Could not detect any working mode")
            return False, None
        
        # Subsequent reads
        if self._mode == 'stream':
            ret, frame = self._cap.read()
        elif self._mode == 'jpg':
            with self._jpg_lock:
                if self._jpg_frame is not None:
                    frame = self._jpg_frame.copy()
                    ret = True
                else:
                    ret = False
                    frame = None
        else:
            ret = False
            frame = None
        
        # Update FPS counter
        now = time.time()
        if now - self._last_time > 0:
            self._fps = 1.0 / (now - self._last_time + 1e-6)
        self._last_time = now
        
        return ret, frame
    
    def set(self, prop_id, value):
        """Set camera property."""
        if self._mode == 'stream' and self._cap:
            self._cap.set(prop_id, value)
    
    def release(self):
        """Release camera resources."""
        if self._mode == 'stream' and self._cap:
            self._cap.release()
        elif self._mode == 'jpg':
            self._stop_event.set()
            if self._jpg_thread:
                self._jpg_thread.join(timeout=1)
