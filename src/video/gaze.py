import numpy as np
import time
import cv2
import mediapipe as mp

class GazeBlinkTracker:
    def __init__(self, blink_threshold=0.20, blink_reset_threshold=0.22, blink_cooldown_sec=0.2):
        self.blink_threshold = blink_threshold
        self.blink_reset_threshold = blink_reset_threshold
        self.blink_cooldown_sec = blink_cooldown_sec

        # State tracking variables
        self.is_blinking = False
        self.blink_count = 0
        self.blink_timestamps = []
        self.last_blink_time = 0

        # MediaPipe Face Mesh refinement indices
        # Right eye: 33 (outer), 133 (inner), 160 (top 1), 144 (bottom 1), 158 (top 2), 153 (bottom 2)
        # Left eye: 263 (outer), 362 (inner), 385 (top 1), 380 (bottom 1), 387 (top 2), 373 (bottom 2)
        self.IDX_R_OUTER = 33
        self.IDX_R_INNER = 133
        self.IDX_R_TOP_1 = 160
        self.IDX_R_BOT_1 = 144
        self.IDX_R_TOP_2 = 158
        self.IDX_R_BOT_2 = 153

        self.IDX_L_OUTER = 263
        self.IDX_L_INNER = 362
        self.IDX_L_TOP_1 = 385
        self.IDX_L_BOT_1 = 380
        self.IDX_L_TOP_2 = 387
        self.IDX_L_BOT_2 = 373

        # Iris centers
        self.IDX_R_IRIS = 468
        self.IDX_L_IRIS = 473

    def _calculate_ear(self, landmarks, outer, inner, top1, bot1, top2, bot2, w, h):
        def lm_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        p_outer = lm_pt(outer)
        p_inner = lm_pt(inner)
        p_top1 = lm_pt(top1)
        p_bot1 = lm_pt(bot1)
        p_top2 = lm_pt(top2)
        p_bot2 = lm_pt(bot2)

        # Vertical distances
        d_v1 = np.linalg.norm(p_top1 - p_bot1)
        d_v2 = np.linalg.norm(p_top2 - p_bot2)

        # Horizontal distance
        d_h = np.linalg.norm(p_outer - p_inner)

        if d_h == 0:
            return 0.0

        ear = (d_v1 + d_v2) / (2.0 * d_h)
        return ear

    def _calculate_gaze_deviation(self, landmarks, outer, inner, iris_idx, w, h):
        def lm_pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        p_outer = lm_pt(outer)
        p_inner = lm_pt(inner)
        p_iris = lm_pt(iris_idx)

        # Eye midpoint
        p_midpoint = (p_outer + p_inner) / 2.0
        eye_width = np.linalg.norm(p_outer - p_inner)

        if eye_width == 0:
            return 0.0

        # Calculate deviation vector relative to eye width
        deviation_vec = (p_iris - p_midpoint) / eye_width

        # We compute the magnitude of horizontal & vertical deviation
        deviation = np.linalg.norm(deviation_vec)
        return deviation

    def process_face(self, face_landmarks, w, h):
        """
        Processes face landmarks to update blink state and calculate gaze deviation.
        Returns a dict of features: {ear, blink_count, blink_rate_bpm, gaze_deviation}
        """
        now = time.time()

        # Calculate EAR for both eyes
        ear_r = self._calculate_ear(face_landmarks, self.IDX_R_OUTER, self.IDX_R_INNER, 
                                    self.IDX_R_TOP_1, self.IDX_R_BOT_1, self.IDX_R_TOP_2, self.IDX_R_BOT_2, w, h)
        ear_l = self._calculate_ear(face_landmarks, self.IDX_L_OUTER, self.IDX_L_INNER, 
                                    self.IDX_L_TOP_1, self.IDX_L_BOT_1, self.IDX_L_TOP_2, self.IDX_L_BOT_2, w, h)
        ear = (ear_r + ear_l) / 2.0

        # Blink state machine with cooldown to avoid double counting
        if ear < self.blink_threshold:
            if not self.is_blinking and (now - self.last_blink_time > self.blink_cooldown_sec):
                self.is_blinking = True
                self.blink_count += 1
                self.blink_timestamps.append(now)
                self.last_blink_time = now
        elif ear > self.blink_reset_threshold:
            self.is_blinking = False

        # Calculate Gaze Deviation (averaged for both eyes)
        gaze_r = self._calculate_gaze_deviation(face_landmarks, self.IDX_R_OUTER, self.IDX_R_INNER, self.IDX_R_IRIS, w, h)
        gaze_l = self._calculate_gaze_deviation(face_landmarks, self.IDX_L_OUTER, self.IDX_L_INNER, self.IDX_L_IRIS, w, h)
        gaze_deviation = (gaze_r + gaze_l) / 2.0

        # Filter out timestamps older than 60 seconds to compute current Blinks per Minute (BPM)
        self.blink_timestamps = [t for t in self.blink_timestamps if now - t <= 60.0]
        blink_rate_bpm = len(self.blink_timestamps)

        return {
            "ear": float(ear),
            "blink_count": int(self.blink_count),
            "blink_rate_bpm": float(blink_rate_bpm),
            "gaze_deviation": float(gaze_deviation)
        }
