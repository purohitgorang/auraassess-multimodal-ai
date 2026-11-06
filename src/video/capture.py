import cv2
import threading
import time

class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.grabbed = False
        self.stopped = False
        self.lock = threading.Lock()

        # Set buffer size to 1 to reduce lag (relevant for webcams)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def run(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stop()
                break

            grabbed, frame = self.cap.read()

            if not grabbed:
                # If reading from a file, wait a bit or stop at EOF
                if not isinstance(self.src, int):
                    self.stopped = True
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame
                self.grabbed = grabbed

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            # Return a copy to prevent race conditions during write/read
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    # Quick test of the capture thread
    print("Starting VideoCaptureThread test...")
    cap_thread = VideoCaptureThread(0)
    cap_thread.start()
    
    time.sleep(2)  # Let it run for 2 seconds
    
    grabbed, frame = cap_thread.read()
    if grabbed:
        print(f"Success! Captured frame of size: {frame.shape}")
    else:
        print("Webcam not available or frame could not be read. (Normal if running headlessly).")
        
    cap_thread.stop()
    cap_thread.join()
    print("Test complete.")
