import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import math

from time_splits import time_splits
from audio import read_wav_file, play_frequencies
from helper_functions import split_into_segments
from fft import add_zero_padding, find_peak_frequencies
import os


WAV_FILE = os.path.join(os.path.dirname(__file__), "Pink_Panther_Music_Box.wav")

# Read WAV file
audio, framerate = read_wav_file(WAV_FILE)
print(f"Loaded '{WAV_FILE}' with {len(audio)} samples at {framerate} Hz")

# Split into segments
segments = split_into_segments(audio, framerate)
print(f"Split audio into {len(segments)} segments based on time_splits.")

# Print the lenght of the longest segment
max_segment_length = max(len(segment) for segment in segments)
print(f"Longest segment length: {max_segment_length} samples")

# Add zero-padding to each segment
min_length = 2 ** int(np.ceil(np.log2(max_segment_length)))
print(f"Zero-padding segments to length: {min_length} samples")
padded_segments = []
for i, segment in enumerate(segments):
    seg_start, seg_end = time_splits[i]
    padded_segment = add_zero_padding(segment, min_length)
    padded_segments.append(padded_segment)

# Use FFT to find peak frequencies
peak_frequencies = find_peak_frequencies(padded_segments, framerate)
print("Identified peak frequencies for each segment.")


# Calculate the note based on the peak frequency
def freq_to_note(freq):
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    note_number = 12 * math.log2(freq / 440) + 49
    note_number = round(note_number)

    note = (note_number - 1) % len(notes)
    note = notes[note]

    octave = (note_number + 8) // len(notes)

    return note, octave


notes = []
for i, freq in enumerate(peak_frequencies):
    note, octave = freq_to_note(freq)
    seg_start, seg_end = time_splits[i]
    print(
        f"Segment {i + 1}: {seg_start}s to {seg_end}s - Peak Frequency: {freq:.2f} Hz - Note: {note}{octave}"
    )
    notes.append((note, octave))


# Plot all peak frequencies in order of segments

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(peak_frequencies) + 1), peak_frequencies, marker="o")
plt.title("Peak Frequencies of Segments")
plt.xlabel("Segment Number")
plt.ylabel("Peak Frequency (Hz)")
plt.grid(True)
plt.xticks(range(1, len(peak_frequencies) + 1))
plt.tight_layout()
plt.show()

# Play the identified peak frequencies as sine waves
play_frequencies(peak_frequencies, duration=0.2)
