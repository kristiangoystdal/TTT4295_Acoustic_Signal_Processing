import sounddevice as sd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from time_splits import time_splits


def split_into_segments(audio, framerate):
    segments = []
    for start, end in time_splits:
        start_idx = max(0, int(start * framerate))
        end_idx = min(len(audio), int(end * framerate))
        if end_idx > start_idx:
            segments.append(audio[start_idx:end_idx])
        else:
            segments.append(np.array([], dtype=audio.dtype))
    return segments


# Calculate the note based on the peak frequency
def freq_to_note(freq):
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    n = round(12 * math.log2(freq / 440.0) + 9)  # semitone index, A4=440 -> n=9
    note_index = n % 12
    octave = n // 12 + 4

    ideal_freq = 440.0 * 2 ** ((n - 9) / 12)

    deviation_cents = 1200 * math.log2(freq / ideal_freq)

    return note_names[note_index], octave, ideal_freq, deviation_cents


# Save notes to a text file
def save_notes_to_file(filename, notes):
    with open(filename, "w") as f:
        for index, note, octave, freq, ideal_freq, deviation_cents in notes:
            f.write(
                f"Segment {index + 1}: {note}{octave} - {freq:.2f} Hz (Ideal: {ideal_freq:.2f} Hz, Deviation: {deviation_cents:.2f} cents)\n"
            )


def plot_waveform(segment, framerate, title="Waveform of Segment", index=None):

    times = np.arange(len(segment)) / framerate

    plt.figure(figsize=(10, 4))
    plt.plot(times, segment)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.xlim(0, times[-1])
    plt.grid()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)
    save_path = os.path.join(
        script_dir,
        "Overleaf",
        "data",
        "time_plots",
        "time_plot_segment_{}.png".format(index if index is not None else "unknown"),
    )
    print("Saving figure to:", save_path)
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def get_saving_path(folder, filename, index=None, filetype="txt"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)
    save_path = os.path.join(
        script_dir,
        "Overleaf",
        "data",
        folder,
        "{}{}.{}".format(filename, index if index is not None else "", filetype),
    )

    return save_path
