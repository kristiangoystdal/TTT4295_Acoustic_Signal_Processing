import numpy as np


def fft_segment(segment, framerate):
    n = len(segment)
    fft_size = 2 ** int(np.ceil(np.log2(n))) 
    fft_result = np.fft.rfft(segment, n=fft_size)
    fft_freqs = np.fft.rfftfreq(fft_size, d=1 / framerate)
    magnitude = np.abs(fft_result)

    # Find peak frequency
    peak_idx = np.argmax(magnitude)
    peak_freq = fft_freqs[peak_idx]

    return peak_freq


def find_peak_frequencies(segments, framerate):
    peak_frequencies = []
    for segment in segments:
        peak_freq = fft_segment(segment, framerate)
        peak_frequencies.append(peak_freq)

    return peak_frequencies


def add_zero_padding(segment, target_length):
    current_length = len(segment)
    if current_length >= target_length:
        return segment
    padding = np.zeros(target_length - current_length, dtype=segment.dtype)
    return np.concatenate((segment, padding))
