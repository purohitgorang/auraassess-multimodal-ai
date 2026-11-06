import numpy as np
import threading
import sounddevice as sd
import soundfile as sf
import time

class AudioCaptureThread:
    def __init__(self, src=None, sample_rate=16000, block_duration_sec=0.1):
        """
        src: None for live microphone, or file path to a WAV/MP3 file.
        sample_rate: sampling rate (default 16000 Hz for speech processing).
        block_duration_sec: duration of each raw callback block.
        """
        self.src = src
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_duration_sec)
        
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.stopped = True
        self.stream = None

        # Thread variables for file playback mode
        self.file_thread = None
        self.file_data = None
        self.file_ptr = 0

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status warning: {status}")
        with self.lock:
            # Append new audio frames to the buffer
            self.buffer = np.append(self.buffer, indata[:, 0])
            
            # Keep buffer size capped at last 30 seconds of audio to prevent memory leaks
            max_len = self.sample_rate * 30
            if len(self.buffer) > max_len:
                self.buffer = self.buffer[-max_len:]

    def _file_reader_loop(self):
        """Simulates mic stream by reading chunks of a WAV file at real-time pace."""
        chunk_size = self.block_size
        while not self.stopped:
            start_time = time.time()
            if self.file_ptr >= len(self.file_data):
                # End of file reached
                self.stopped = True
                break
                
            chunk = self.file_data[self.file_ptr : self.file_ptr + chunk_size]
            self.file_ptr += chunk_size
            
            # If the last chunk is smaller than expected, pad it
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            with self.lock:
                self.buffer = np.append(self.buffer, chunk)
                max_len = self.sample_rate * 30
                if len(self.buffer) > max_len:
                    self.buffer = self.buffer[-max_len:]

            # Sleep to match real-time playback
            elapsed = time.time() - start_time
            sleep_time = max(0.0, (self.block_size / self.sample_rate) - elapsed)
            time.sleep(sleep_time)

    def start(self):
        self.stopped = False
        if self.src is None:
            # Live mic recording mode
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.block_size,
                callback=self._audio_callback,
                dtype='float32'
            )
            self.stream.start()
        else:
            # Read and play back a file in a simulated real-time stream
            data, fs = sf.read(self.src, dtype='float32')
            # Resample or extract first channel if stereo
            if len(data.shape) > 1:
                data = data[:, 0]
            if fs != self.sample_rate:
                # Basic resampling if soundfile doesn't match
                # (For high quality, use librosa, but this is a fallback)
                import scipy.signal
                num_samples = int(len(data) * self.sample_rate / fs)
                data = scipy.signal.resample(data, num_samples)
            
            self.file_data = data
            self.file_ptr = 0
            self.file_thread = threading.Thread(target=self._file_reader_loop)
            self.file_thread.start()

    def get_audio_chunk(self, duration_sec=3.0):
        """Returns the last N seconds of audio in the buffer."""
        num_samples = int(self.sample_rate * duration_sec)
        with self.lock:
            if len(self.buffer) == 0:
                return np.zeros(num_samples, dtype=np.float32)
            
            if len(self.buffer) < num_samples:
                # Pad with zeros if buffer is smaller than requested duration
                return np.pad(self.buffer, (num_samples - len(self.buffer), 0))
            
            return self.buffer[-num_samples:].copy()

    def stop(self):
        self.stopped = True
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.file_thread is not None:
            self.file_thread.join()
            self.file_thread = None

if __name__ == "__main__":
    print("Testing AudioCaptureThread (Microphone)...")
    try:
        audio_thread = AudioCaptureThread()
        audio_thread.start()
        print("Recording for 3 seconds...")
        time.sleep(3)
        chunk = audio_thread.get_audio_chunk(3.0)
        print(f"Success! Captured audio chunk of shape: {chunk.shape}, max amplitude: {np.max(np.abs(chunk))}")
        audio_thread.stop()
    except Exception as e:
        print(f"Mic not available or error occurred: {e} (Normal if running headlessly).")
