import numpy as np
import librosa

class AcousticProsodyAnalyzer:
    def __init__(self, sample_rate=16000, fmin=60, fmax=300, energy_threshold=0.01):
        """
        sample_rate: sampling rate of input audio.
        fmin: minimum frequency to track (typically 60-80Hz for adults).
        fmax: maximum frequency to track (typically 250-300Hz for adult speech).
        energy_threshold: RMS amplitude threshold to classify frames as voiced/silent.
        """
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.energy_threshold = energy_threshold

    def extract_features(self, audio_chunk):
        """
        Extracts vocal prosody features: pitch mean/variance and voice energy variance.
        Returns a dict of features.
        """
        # If the chunk is empty or all zeros, return default values
        if len(audio_chunk) == 0 or np.all(audio_chunk == 0):
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "energy_mean": 0.0,
                "energy_std": 0.0,
                "is_speaking": False
            }

        # 1. Calculate Root Mean Square (RMS) energy per frame
        # frame_length=512 samples corresponds to 32ms at 16kHz
        rms = librosa.feature.rms(y=audio_chunk, frame_length=512, hop_length=256)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        is_speaking = float(energy_mean) > self.energy_threshold

        # 2. Extract pitch (F0) using the YIN algorithm
        voiced_f0 = []
        try:
            if is_speaking:
                # Track fundamental frequency (F0)
                f0 = librosa.yin(
                    y=audio_chunk,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    sr=self.sample_rate,
                    frame_length=512,
                    hop_length=256
                )
                
                # Filter f0 values: only keep values where RMS energy is above the threshold
                # to prevent background hums or silent artifacts from polluting the pitch metric
                voiced_f0 = f0[rms > self.energy_threshold]
                # Filter out NaN/inf values if any
                voiced_f0 = voiced_f0[np.isfinite(voiced_f0)]
        except Exception as e:
            # Handle rare exceptions (e.g. audio too short or empty signal)
            pass

        # 3. Calculate pitch stats
        if len(voiced_f0) > 2:
            pitch_mean = np.mean(voiced_f0)
            pitch_std = np.std(voiced_f0)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0

        return {
            "pitch_mean": float(pitch_mean),
            "pitch_std": float(pitch_std),
            "energy_mean": float(energy_mean),
            "energy_std": float(energy_std),
            "is_speaking": bool(is_speaking)
        }

if __name__ == "__main__":
    print("Testing AcousticProsodyAnalyzer...")
    analyzer = AcousticProsodyAnalyzer()
    
    # Test on a dummy synthetic sine wave (pitch = 150 Hz)
    sr = 16000
    t = np.linspace(0, 3, sr * 3, endpoint=False) # 3 seconds
    sine_wave = 0.1 * np.sin(2 * np.pi * 150.0 * t)
    
    res = analyzer.extract_features(sine_wave)
    print(f"Extraction result on 150Hz sine wave: {res}")
