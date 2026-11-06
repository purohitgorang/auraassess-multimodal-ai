import numpy as np
import time

class SpeechLatencyTracker:
    def __init__(self, energy_threshold=0.015, min_speech_frames=3, frame_duration_sec=0.1):
        """
        energy_threshold: RMS amplitude threshold above which audio is considered speech.
        min_speech_frames: number of consecutive frames above threshold needed to confirm speech start.
        frame_duration_sec: duration of each incoming audio frame (block size in seconds).
        """
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames
        self.frame_duration = frame_duration_sec

        # State management
        self.prompt_ended_time = None
        self.listening = False
        self.consecutive_speech_frames = 0
        self.latency = 0.0

    def trigger_prompt_end(self):
        """
        Call this function the moment the interviewer prompt/question finishes playing.
        """
        self.prompt_ended_time = time.time()
        self.listening = True
        self.consecutive_speech_frames = 0
        self.latency = 0.0

    def process_audio_block(self, audio_block):
        """
        Processes a short raw audio block (e.g., 100ms) to detect candidate speech start.
        Returns:
            latency: float or None (returns the calculated latency the instant speech is detected, otherwise None).
        """
        if not self.listening or self.prompt_ended_time is None:
            return None

        # Calculate RMS energy of the current audio block
        if len(audio_block) == 0:
            return None
            
        rms = np.sqrt(np.mean(audio_block ** 2))

        if rms > self.energy_threshold:
            self.consecutive_speech_frames += 1
        else:
            self.consecutive_speech_frames = 0  # Reset on silence/interruption

        # Verify speech is sustained (filtering out clicks/pops)
        if self.consecutive_speech_frames >= self.min_speech_frames:
            now = time.time()
            # Calculate total time elapsed since prompt end
            # Subtract the duration of the speech confirmation window to get the true start boundary
            self.latency = (now - self.prompt_ended_time) - (self.min_speech_frames * self.frame_duration)
            self.latency = max(0.0, self.latency)  # Ensure positive value
            
            # Reset listening state until next prompt trigger
            self.listening = False
            self.prompt_ended_time = None
            
            return self.latency

        return None

if __name__ == "__main__":
    print("Testing SpeechLatencyTracker...")
    tracker = SpeechLatencyTracker(energy_threshold=0.01)
    
    # Simulate a prompt ending now
    tracker.trigger_prompt_end()
    print("Prompt ended. Simulating silence for 1.5 seconds...")
    time.sleep(1.5)
    
    # Send empty/silent audio blocks
    empty_block = np.zeros(1600)  # 100ms at 16kHz
    for _ in range(5):
        lat = tracker.process_audio_block(empty_block)
        assert lat is None, "Speech detected prematurely on silent frame!"

    print("Simulating active candidate speech...")
    speech_block = 0.05 * np.random.randn(1600)  # 100ms loud noise
    
    # Feed speech blocks to trigger detection
    triggered = False
    for i in range(5):
        lat = tracker.process_audio_block(speech_block)
        if lat is not None:
            print(f"Speech start detected! Computed Latency: {lat:.3f} seconds (Expected ~1.5s).")
            triggered = True
            break
            
    if not triggered:
        print("Failed to trigger speech start.")
