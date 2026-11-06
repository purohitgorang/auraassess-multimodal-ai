import pytest
import numpy as np
from src.video.face_features import FaceFeatureExtractor
from src.video.pose_features import PoseFeatureExtractor
from src.video.gaze import GazeBlinkTracker

def test_face_features_empty_frame():
    extractor = FaceFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = extractor.extract_features(dummy_frame)
    assert res is None, "Extractor should return None when no face is present"
    extractor.close()

def test_pose_features_empty_frame():
    extractor = PoseFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = extractor.extract_features(dummy_frame)
    assert res is None, "Extractor should return None when no body is present"
    extractor.close()

def test_gaze_blink_tracker_ear():
    tracker = GazeBlinkTracker()
    
    # Mock landmarker structures
    class MockLandmark:
        def __init__(self, x, y, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    # Construct mock eye landmarks with a known aspect ratio
    # Horizontal: 33 to 133, distance = 0.1
    # Vertical: 160 to 144, distance = 0.02
    # Vertical 2: 158 to 153, distance = 0.02
    landmarks = [None] * 500
    
    # Right Eye
    landmarks[33] = MockLandmark(0.1, 0.5)
    landmarks[133] = MockLandmark(0.2, 0.5)
    landmarks[160] = MockLandmark(0.15, 0.491)
    landmarks[144] = MockLandmark(0.15, 0.509)
    landmarks[158] = MockLandmark(0.16, 0.491)
    landmarks[153] = MockLandmark(0.16, 0.509)

    # Left Eye (Identical coordinates)
    landmarks[263] = MockLandmark(0.4, 0.5)
    landmarks[362] = MockLandmark(0.5, 0.5)
    landmarks[385] = MockLandmark(0.45, 0.491)
    landmarks[380] = MockLandmark(0.45, 0.509)
    landmarks[387] = MockLandmark(0.46, 0.491)
    landmarks[373] = MockLandmark(0.46, 0.509)

    # Iris
    landmarks[468] = MockLandmark(0.15, 0.5)
    landmarks[473] = MockLandmark(0.45, 0.5)

    res = tracker.process_face(landmarks, 100, 100)
    
    # Vertical distance = 1.8 pixels, Horizontal = 10 pixels
    # EAR = (1.8 + 1.8) / (2 * 10) = 3.6 / 20 = 0.18
    assert abs(res["ear"] - 0.18) < 0.01
    assert res["blink_count"] == 1, "Should record 1 blink when EAR drops to threshold"

