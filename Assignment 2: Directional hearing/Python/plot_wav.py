import numpy as np
import wave

import matplotlib.pyplot as plt

WAV_FILE = "Assignment 1: Music Box/Pink_Panther_Music_Box.wav"  # Change to your actual file name

with wave.open(WAV_FILE, "rb") as wf:
    n_channels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    n_frames = wf.getnframes()
    audio_data = wf.readframes(n_frames)

# Convert bytes to numpy array
dtype = np.int16 if sampwidth == 2 else np.uint8
audio = np.frombuffer(audio_data, dtype=dtype)

# If stereo, take one channel
if n_channels > 1:
    audio = audio[::n_channels]

time = np.linspace(0, n_frames / framerate, num=n_frames)

min_time_display = 0  # seconds
max_time_display = min_time_display + 270  # seconds

plt.figure(figsize=(10, 4))
plt.plot(time, audio)
plt.title(f"Waveform of {WAV_FILE}")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.xlim(min_time_display, max_time_display)
plt.tight_layout()
plt.show()
