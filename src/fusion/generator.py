import numpy as np
import pandas as pd
import os

class SyntheticInterviewGenerator:
    def __init__(self, session_duration_sec=60, sample_rate_hz=1):
        self.duration = session_duration_sec
        self.hz = sample_rate_hz
        self.steps = session_duration_sec * sample_rate_hz

    def generate_session(self, state, session_id):
        """
        Generates a second-by-second log for a single session given a target state.
        Profiles:
        - Stress: low furrow_z, high sway_z, high blink (ear_z drops), self-touch active, high latency.
        - Chill: moderate furrow_z, low sway_z, stable ear, self-touch inactive, moderate latency.
        - Confident: moderate furrow_z, low sway_z, stable ear, low gaze_dev, prompt latency.
        - Overconfident: raise_z high, sway_z low, forward_lean negative (leaning back), energy_z high, latency near zero.
        """
        rows = []
        for step in range(self.steps):
            timestamp = float(step / self.hz)
            
            # Default baselines (neutral)
            furrow_z = np.random.normal(0.0, 0.5)
            raise_z = np.random.normal(0.0, 0.5)
            ear = np.random.normal(0.25, 0.02)
            ear_z = np.random.normal(0.0, 0.5)
            sway_score = np.random.normal(0.01, 0.002)
            sway_score_z = np.random.normal(0.0, 0.5)
            lateral_lean = np.random.normal(0.0, 1.0)
            forward_lean = np.random.normal(0.0, 0.1)
            pacifying_active = 1.0 if np.random.random() < 0.05 else 0.0
            
            # New body features defaults
            hand_velocity_z = np.random.normal(0.0, 0.5)
            gesture_score_z = np.random.normal(0.0, 0.5)
            crossed_arms = 0.0

            # Default audio features
            pitch_mean_z = np.random.normal(0.0, 0.5)
            energy_mean_z = np.random.normal(0.0, 0.5)
            pitch_mean = 120.0 + pitch_mean_z * 15.0
            pitch_std = np.random.normal(5.0, 1.0)
            energy_mean = 0.05 + energy_mean_z * 0.01
            energy_std = np.random.normal(0.01, 0.002)
            speech_latency = np.random.normal(1.5, 0.3)

            # Apply profile-specific deviations
            if state == "Stress":
                furrow_z = np.random.normal(-2.2, 0.6)  # Brows furrowed
                sway_score_z = np.random.normal(1.8, 0.5)  # Fidgeting
                sway_score = 0.05 + 0.01 * np.random.randn()
                ear = np.random.normal(0.20, 0.03)  # Rapid blinking/half-closed eyes
                ear_z = np.random.normal(-1.8, 0.5)
                pacifying_active = 1.0 if np.random.random() < 0.60 else 0.0  # High neck touch probability
                
                # Dynamic body upgrades
                hand_velocity_z = np.random.normal(2.2, 0.5)  # High nervous hand shaking/fidgeting
                gesture_score_z = np.random.normal(-1.2, 0.4)  # Hands tucked in/withdrawn
                crossed_arms = 1.0 if np.random.random() < 0.45 else 0.0  # Frequent crossed arms

                # Audio deviations
                pitch_mean_z = np.random.normal(1.2, 0.5)  # Shaky/stressed higher pitch
                energy_mean_z = np.random.normal(-0.8, 0.4) # Soft/muffled speech
                pitch_std = np.random.normal(2.0, 0.5)      # Monotone / lack of prosody
                speech_latency = np.random.normal(3.8, 0.8)  # High latency/hesitant

            elif state == "Chill":
                sway_score_z = np.random.normal(-0.8, 0.3)  # Very still
                sway_score = 0.005 + 0.001 * np.random.randn()
                pacifying_active = 1.0 if np.random.random() < 0.02 else 0.0  # Rare touching
                
                hand_velocity_z = np.random.normal(-0.6, 0.2)  # Still hands
                gesture_score_z = np.random.normal(0.0, 0.3)  # Normal hand space

                # Audio deviations
                pitch_mean_z = np.random.normal(-0.2, 0.3)
                energy_mean_z = np.random.normal(-0.1, 0.3)
                speech_latency = np.random.normal(1.8, 0.4)
                
            elif state == "Confident":
                gaze_dev = np.random.normal(0.07, 0.02)  # Looking at camera
                sway_score_z = np.random.normal(-0.6, 0.3)
                sway_score = 0.007 + 0.001 * np.random.randn()
                pacifying_active = 0.0
                
                # Confident gestures (expressive hand gestures)
                hand_velocity_z = np.random.normal(0.4, 0.2)  # Fluid hand moves
                gesture_score_z = np.random.normal(1.4, 0.4)  # Open gestures

                # Audio deviations
                pitch_mean_z = np.random.normal(0.1, 0.3)
                energy_mean_z = np.random.normal(0.6, 0.3)
                speech_latency = np.random.normal(0.8, 0.2)
                
            elif state == "Overconfident":
                forward_lean = np.random.normal(-0.4, 0.1)  # Leaning back (negative)
                raise_z = np.random.normal(1.8, 0.5)  # Raised brows
                pacifying_active = 0.0
                
                # Exaggerated space-taking hand moves
                hand_velocity_z = np.random.normal(1.2, 0.4)
                gesture_score_z = np.random.normal(2.3, 0.5)

                # Audio deviations
                pitch_mean_z = np.random.normal(1.6, 0.4)   # Loud / dominant pitch
                energy_mean_z = np.random.normal(1.8, 0.4)  # High voice intensity
                speech_latency = np.random.normal(0.2, 0.1)  # Immediate responses

            # Re-verify EAR is clipped to physical limits
            ear = max(0.05, min(0.40, ear))
            pitch_mean = max(50.0, pitch_mean)
            energy_mean = max(0.001, energy_mean)

            # Generate other frame-level base values to compile rows (Multimodal)
            row = {
                "session_id": session_id,
                "timestamp": timestamp,
                "is_calibrating": False,
                # Video metrics
                "furrow_index": float(0.5 + furrow_z * 0.05),
                "raise_index": float(0.1 + raise_z * 0.03),
                "ear": float(ear),
                "sway_score": float(sway_score),
                "lateral_lean": float(lateral_lean),
                "forward_lean": float(forward_lean),
                "pacifying_active": float(pacifying_active),
                "gaze_deviation": float(np.random.normal(0.1, 0.05) if state != "Confident" else 0.07),
                "hand_velocity": float(0.02 + (hand_velocity_z + 1.0) * 0.01),
                "gesture_score": float(0.45 + gesture_score_z * 0.1),
                "crossed_arms": float(crossed_arms),
                # Audio metrics
                "pitch_mean": float(pitch_mean),
                "pitch_std": float(pitch_std),
                "energy_mean": float(energy_mean),
                "energy_std": float(energy_std),
                "speech_latency": float(speech_latency),
                # Normalization Z-scores
                "furrow_index_z": float(furrow_z),
                "raise_index_z": float(raise_z),
                "ear_z": float(ear_z),
                "sway_score_z": float(sway_score_z),
                "hand_velocity_z": float(hand_velocity_z),
                "gesture_score_z": float(gesture_score_z),
                "crossed_arms_z": float(crossed_arms),
                "pitch_mean_z": float(pitch_mean_z),
                "energy_mean_z": float(energy_mean_z),
                # Target Label
                "label": state
            }
            rows.append(row)
        return rows

    def generate_dataset(self, output_csv="synthetic_interview_data.csv", num_sessions_per_state=50):
        states = ["Stress", "Chill", "Confident", "Overconfident"]
        dataset = []

        session_counter = 0
        for state in states:
            for _ in range(num_sessions_per_state):
                session_rows = self.generate_session(state, session_counter)
                dataset.extend(session_rows)
                session_counter += 1

        df = pd.DataFrame(dataset)
        df.to_csv(output_csv, index=False)
        print(f"Generated synthetic dataset containing {len(df)} rows across {session_counter} sessions.")
        print(f"Dataset successfully written to: {output_csv}")

if __name__ == "__main__":
    generator = SyntheticInterviewGenerator()
    generator.generate_dataset()
