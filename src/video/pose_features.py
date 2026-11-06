import cv2
import numpy as np
import mediapipe as mp
from collections import deque

class PoseFeatureExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, sway_window_len=150):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(


            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark Indices (MediaPipe Pose specs)
        self.IDX_NOSE = 0
        self.IDX_L_SHOULDER = 11
        self.IDX_R_SHOULDER = 12
        self.IDX_L_WRIST = 15
        self.IDX_R_WRIST = 16
        self.IDX_L_HIP = 23
        self.IDX_R_HIP = 24

        # Sliding window for postural sway (x_torso coordinate history)
        self.torso_x_history = deque(maxlen=sway_window_len)

        # Previous frame wrist history (for hand velocity)
        self.prev_l_wrist = None
        self.prev_r_wrist = None

    def extract_features(self, frame):
        """
        Extracts skeletal features: postural sway, lean angles, self-touch events,
        hand velocity, gesture score, and crossed arms posture.
        Returns a dict of features, or None if no pose is detected.
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark

        # Convert landmarks to pixel coordinates (3D)
        def lm_to_arr(lm_idx):
            lm = landmarks[lm_idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])

        P_nose = lm_to_arr(self.IDX_NOSE)
        P_l_shoulder = lm_to_arr(self.IDX_L_SHOULDER)
        P_r_shoulder = lm_to_arr(self.IDX_R_SHOULDER)
        P_l_wrist = lm_to_arr(self.IDX_L_WRIST)
        P_r_wrist = lm_to_arr(self.IDX_R_WRIST)
        P_l_hip = lm_to_arr(self.IDX_L_HIP)
        P_r_hip = lm_to_arr(self.IDX_R_HIP)

        # 1. Scale Normalizer (Shoulder Width)
        D_shoulder_width = np.linalg.norm(P_l_shoulder - P_r_shoulder)
        if D_shoulder_width == 0:
            D_shoulder_width = 1.0  # Prevent division by zero

        # 2. Neck Coordinate (midpoint between shoulders)
        P_neck = (P_l_shoulder + P_r_shoulder) / 2.0
        P_torso_mid = P_neck  # midpoint torso

        # 3. Pacifying touch (Self-touching hands-to-face or hands-to-neck)
        dist_r_wrist_face = np.linalg.norm(P_r_wrist - P_nose) / D_shoulder_width
        dist_l_wrist_face = np.linalg.norm(P_l_wrist - P_nose) / D_shoulder_width
        dist_r_wrist_neck = np.linalg.norm(P_r_wrist - P_neck) / D_shoulder_width
        dist_l_wrist_neck = np.linalg.norm(P_l_wrist - P_neck) / D_shoulder_width

        THRES_FACE = 0.45
        THRES_NECK = 0.40

        pacifying_active = (
            dist_r_wrist_face < THRES_FACE or
            dist_l_wrist_face < THRES_FACE or
            dist_r_wrist_neck < THRES_NECK or
            dist_l_wrist_neck < THRES_NECK
        )

        # 4. Lean Angle (lateral spine angle with vertical)
        P_hip_mid = (P_l_hip + P_r_hip) / 2.0
        dx = P_torso_mid[0] - P_hip_mid[0]
        dy = P_hip_mid[1] - P_torso_mid[1]
        lateral_lean_deg = np.degrees(np.arctan2(dx, dy))

        # Forward leaning (Z depth difference between shoulder and hips)
        forward_lean_score = (P_hip_mid[2] - P_torso_mid[2]) / D_shoulder_width

        # 5. Postural Sway (Rolling variance of torso X coordinate)
        self.torso_x_history.append(P_torso_mid[0] / D_shoulder_width)
        sway_score = 0.0
        if len(self.torso_x_history) >= 30:
            sway_score = np.var(self.torso_x_history)

        # 6. Hand Velocity (Fidget speed)
        hand_velocity = 0.0
        if self.prev_l_wrist is not None and self.prev_r_wrist is not None:
            v_l = np.linalg.norm(P_l_wrist - self.prev_l_wrist)
            v_r = np.linalg.norm(P_r_wrist - self.prev_r_wrist)
            hand_velocity = (v_l + v_r) / (2.0 * D_shoulder_width)

        self.prev_l_wrist = P_l_wrist.copy()
        self.prev_r_wrist = P_r_wrist.copy()

        # 7. Gesture Expressiveness Score (distance of hands from chest center)
        dist_l_gesture = np.linalg.norm(P_l_wrist - P_torso_mid)
        dist_r_gesture = np.linalg.norm(P_r_wrist - P_torso_mid)
        gesture_score = (dist_l_gesture + dist_r_gesture) / (2.0 * D_shoulder_width)

        # 8. Crossed-Arms Detection
        # Check if left wrist is to the right of the right shoulder, and right wrist is to the left of the left shoulder,
        # and vertical position is in the chest/belly area (between shoulders and hips)
        y_bounds = (
            P_neck[1] < P_l_wrist[1] < P_l_hip[1] and 
            P_neck[1] < P_r_wrist[1] < P_r_hip[1]
        )
        # MediaPipe X coordinates: smaller values are candidate's left (our right if mirrored),
        # but in raw normalized 0-1 coordinates, X increases left-to-right on image.
        # Right shoulder (R_SHOULDER) has SMALLER X on screen than Left shoulder (L_SHOULDER).
        # Wrist cross midline: Left Wrist X < Right Shoulder X, and Right Wrist X > Left Shoulder X.
        x_cross = (
            P_l_wrist[0] < P_r_shoulder[0] + 0.15 * D_shoulder_width and 
            P_r_wrist[0] > P_l_shoulder[0] - 0.15 * D_shoulder_width
        )
        crossed_arms = bool(y_bounds and x_cross)

        return {
            "pacifying_active": bool(pacifying_active),
            "lateral_lean": float(lateral_lean_deg),
            "forward_lean": float(forward_lean_score),
            "sway_score": float(sway_score),
            "hand_velocity": float(hand_velocity),
            "gesture_score": float(gesture_score),
            "crossed_arms": float(crossed_arms),
            "shoulder_width": float(D_shoulder_width)
        }


    def close(self):
        self.pose.close()

if __name__ == "__main__":
    print("Testing PoseFeatureExtractor...")
    extractor = PoseFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = extractor.extract_features(dummy_frame)
    print(f"Extraction result on blank frame: {res} (Expected: None)")
    extractor.close()
