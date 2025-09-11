import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import math

from time_splits import time_splits
from audio import *
from helper_functions import *
from fft import *
import os


WAV_FILE = os.path.join(os.path.dirname(__file__), "Pink_Panther_Music_Box.wav")

# Read WAV file
audio, framerate = read_wav_file(WAV_FILE)
print(f"Loaded '{WAV_FILE}' with {len(audio)} samples at {framerate} Hz")

# Split into segments
segments = split_into_segments(audio, framerate)
print(f"Split audio into {len(segments)} segments based on time_splits.")

# Plot each segment's waveform
for i, segment in enumerate(segments):
    plot_waveform(segment, framerate, title=f"Waveform of Segment {i + 1}", index=i + 1)

# Print the length of the longest segment
max_segment_length = max(len(segment) for segment in segments)
print(f"Longest segment length: {max_segment_length} samples")

# Add zero-padding to each segment
min_length = 2 ** int(np.ceil(np.log2(max_segment_length)))
print(f"Zero-padding segments to length: {min_length} samples")
padded_segments = []
for i, segment in enumerate(segments):
    padded_segment = add_zero_padding(segment, min_length)
    padded_segments.append(padded_segment)

# Compute FFT for each padded segment
fft_segments = []
for segment in padded_segments:
    fft_freqs, magnitude = fft_segment(segment, framerate)
    fft_segments.append((fft_freqs, magnitude))

# Use FFT to find peak frequencies
peak_frequencies = find_peak_frequencies(fft_segments)
print("Identified peak frequencies for each segment.")

# Find harmonic peaks for each segment
# harmonic_peaks = []
# for i, (fft_freqs, magnitude) in enumerate(fft_segments):
#     peak_freq = peak_frequencies[i]
#     peaks = find_harmonic_peaks(fft_freqs, magnitude, peak_freq=peak_freq)
#     harmonic_peaks.append(peaks)

#     # Plot FFT with harmonic peaks
#     plot_fft(
#         fft_freqs,
#         magnitude,
#         title=f"FFT Magnitude Spectrum for Note {i + 1}",
#         harmonic_peaks=peaks,
#         xlim=8000,
#         index=i + 1,
#     )

# Separate melody and bass frequencies
melody_freq = []
melody_notes = []
bass_freq = []
bass_notes = []

for i, freq in enumerate(peak_frequencies):
    note, octave, ideal_freq, deviation_cents = freq_to_note(freq)

    if freq >= 261.63:
        melody_freq.append(freq)
        melody_notes.append((i, note, octave, freq, ideal_freq, deviation_cents))
    else:
        bass_freq.append(freq)
        bass_notes.append((i, note, octave, freq, ideal_freq, deviation_cents))


# Save melody and bass notes to text files
melody_notes_path = get_saving_path("text", "melody_notes.txt")
bass_notes_path = get_saving_path("text", "bass_notes.txt")
save_notes_to_file(melody_notes_path, melody_notes)
save_notes_to_file(bass_notes_path, bass_notes)
print("Saved melody notes to 'melody_notes.txt'")
print("Saved bass notes to 'bass_notes.txt'")

# Plot all peak frequencies in order of segments
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(peak_frequencies) + 1), peak_frequencies, marker="o")
plt.title("Peak Frequencies of Segments")
plt.xlabel("Segment Number")
plt.ylabel("Peak Frequency (Hz)")
plt.grid(True)
plt.xticks(range(1, len(peak_frequencies) + 1))
plt.tight_layout()
# plt.show()

# Play the identified peak frequencies as sine waves
# play_frequencies(melody_freq, duration=0.5)
