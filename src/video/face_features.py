import cv2
import numpy as np
import mediapipe as mp

class FaceFeatureExtractor:
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(


            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark Indices (MediaPipe Face Mesh specs)
        self.IDX_R_EYE_OUTER = 33
        self.IDX_L_EYE_OUTER = 263
        self.IDX_R_EYEBROW_INNER = 70
        self.IDX_L_EYEBROW_INNER = 300
        self.IDX_R_EYE_TOP = 159
        self.IDX_L_EYE_TOP = 386

    def extract_features(self, frame):
        """
        Extracts face geometry features relative to forehead tension (furrowing & raising).
        Returns a dict of features, or None if no face is detected.
        """
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        # Process first detected face
        landmarks = results.multi_face_landmarks[0].landmark

        # Helper to convert landmark normalized coordinates to pixel numpy arrays (3D)
        def lm_to_arr(lm_idx):
            lm = landmarks[lm_idx]
            return np.array([lm.x * w, lm.y * h, lm.z * w])

        P_r_eye_outer = lm_to_arr(self.IDX_R_EYE_OUTER)
        P_l_eye_outer = lm_to_arr(self.IDX_L_EYE_OUTER)
        P_r_brow_inner = lm_to_arr(self.IDX_R_EYEBROW_INNER)
        P_l_brow_inner = lm_to_arr(self.IDX_L_EYEBROW_INNER)
        P_r_eye_top = lm_to_arr(self.IDX_R_EYE_TOP)
        P_l_eye_top = lm_to_arr(self.IDX_L_EYE_TOP)

        # 1. Scale Normalizer (Eye outer-to-outer distance)
        D_scale = np.linalg.norm(P_r_eye_outer - P_l_eye_outer)
        if D_scale == 0:
            D_scale = 1.0  # Prevent division by zero

        # 2. Eyebrow Furrow Index (Distance between inner eyebrows)
        D_brow_inner = np.linalg.norm(P_r_brow_inner - P_l_brow_inner)
        furrow_index = D_brow_inner / D_scale

        # 3. Eyebrow Raising Index (Vertical height of inner brows relative to eye tops)
        # Using y-coordinates directly (MediaPipe y is inverted: top is 0, bottom is 1)
        # Therefore, brow_y - eye_top_y is the distance (since brow is above eye, brow_y is LESS than eye_top_y)
        # Vertical height = Eye_top_y - Brow_inner_y (in pixels)
        H_r_brow = P_r_eye_top[1] - P_r_brow_inner[1]
        H_l_brow = P_l_eye_top[1] - P_l_brow_inner[1]
        raise_index = ((H_r_brow + H_l_brow) / 2.0) / D_scale

        return {
            "furrow_index": float(furrow_index),
            "raise_index": float(raise_index),
            "scale_factor": float(D_scale)
        }

    def close(self):
        self.face_mesh.close()

if __name__ == "__main__":
    # Test on a dummy black image
    print("Testing FaceFeatureExtractor on dummy frame...")
    extractor = FaceFeatureExtractor()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # It should return None since there is no face
    res = extractor.extract_features(dummy_frame)
    print(f"Extraction result on blank frame: {res} (Expected: None)")
    extractor.close()
