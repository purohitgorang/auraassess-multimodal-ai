import pytest
import numpy as np
from src.fusion.calibration import CalibrationEngine
from src.fusion.aggregator import MultimodalAggregator
from src.model.predictor import BehaviorOutcomePredictor

def test_calibration_and_normalization():
    # Set 2 seconds calibration
    calibrator = CalibrationEngine(calibration_duration_sec=2.0)
    calibrator.start(100.0)
    
    # Feed constant calibration data
    for t in np.linspace(100.0, 102.5, 10):
        raw = {
            "furrow_index": 0.5,
            "raise_index": 0.1,
            "ear": 0.25,
            "sway_score": 0.01,
            "hand_velocity": 0.15,
            "gesture_score": 1.10,
            "pitch_mean": 120.0,
            "energy_mean": 0.05
        }
        calibrator.update_calibration(t, raw)
        
    assert calibrator.calibrated is True
    assert calibrator.baselines["gesture_score"]["mean"] == 1.10
    
    # Normalize a stressed state (low furrow, high sway, high hand velocity, high pitch)
    stress_raw = {
        "furrow_index": 0.3, # lower (eyebrows closer)
        "raise_index": 0.1,
        "ear": 0.25,
        "sway_score": 0.05, # higher sway
        "hand_velocity": 0.85, # higher fidgeting
        "gesture_score": 0.40,  # lower (withdrawn hands)
        "pitch_mean": 150.0,  # higher pitch
        "energy_mean": 0.02   # lower energy
    }
    
    z_scores = calibrator.normalize(stress_raw)
    
    assert z_scores["furrow_index_z"] < 0
    assert z_scores["sway_score_z"] > 0
    assert z_scores["hand_velocity_z"] > 0
    assert z_scores["gesture_score_z"] < 0
    assert z_scores["pitch_mean_z"] > 0
    assert z_scores["energy_mean_z"] < 0

def test_aggregator_fusion():
    calibrator = CalibrationEngine(calibration_duration_sec=2.0)
    aggregator = MultimodalAggregator(calibrator)
    
    # Mock video frames
    aggregator.add_video_frame({"ear": 0.26, "furrow_index": 0.5, "sway_score": 0.01, "hand_velocity": 0.1, "gesture_score": 1.0, "crossed_arms": 0.0})
    aggregator.add_video_frame({"ear": 0.24, "furrow_index": 0.5, "sway_score": 0.01, "hand_velocity": 0.1, "gesture_score": 1.0, "crossed_arms": 1.0})
    
    row = aggregator.aggregate_second(1.0)
    
    assert row["ear"] == 0.25  # Average of 0.26 and 0.24
    assert row["crossed_arms"] == 1.0  # logical OR aggregated for boolean
    assert row["is_calibrating"] is True

def test_predictor_heuristics():
    predictor = BehaviorOutcomePredictor()
    
    # Construct calm vector -> Chill/Confident
    chill_vector = {
        "furrow_index_z": 0.0,
        "raise_index_z": 0.0,
        "sway_score_z": 0.0,
        "ear": 0.25,
        "pacifying_active": 0.0,
        "gaze_deviation": 0.08,
        "hand_velocity_z": -0.2,
        "gesture_score_z": 0.1,
        "crossed_arms": 0.0
    }
    
    state, probs = predictor.predict_heuristic(chill_vector)
    assert state in ["Chill", "Confident"]
    
    # Construct stressed vector
    stress_vector = {
        "furrow_index_z": -2.5,
        "raise_index_z": 0.0,
        "sway_score_z": 2.1,
        "ear": 0.16,
        "pacifying_active": 1.0,
        "gaze_deviation": 0.22,
        "hand_velocity_z": 1.8,
        "gesture_score_z": -1.2,
        "crossed_arms": 1.0
    }
    
    state, probs = predictor.predict_heuristic(stress_vector)
    assert state == "Stress"
