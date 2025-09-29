import numpy as np
import os
import matplotlib.pyplot as plt


def fft_segment(signal, framerate):
    n = len(signal)
    fft_size = 2 ** int(np.ceil(np.log2(n)))
    fft_result = np.fft.rfft(signal, n=fft_size)
    fft_freqs = np.fft.rfftfreq(fft_size, d=1 / framerate)
    magnitude = np.abs(fft_result)

    return fft_freqs, magnitude


def find_peak_frequencies(fft_segments):
    peak_frequencies = []
    for fft_freqs, magnitude in fft_segments:
        peak_index = np.argmax(magnitude)
        peak_frequency = fft_freqs[peak_index]
        peak_frequencies.append(peak_frequency)

    return peak_frequencies


def add_zero_padding(segment, target_length):
    current_length = len(segment)
    if current_length >= target_length:
        return segment
    padding = np.zeros(target_length - current_length, dtype=segment.dtype)
    return np.concatenate((segment, padding))


def plot_fft(
    fft_freqs,
    magnitude,
    title="FFT Magnitude Spectrum",
    harmonic_peaks=None,
    xlim=8000,
    index=None,
):
    # Plot the FFT magnitude spectrum with the highest peak as 0 dB
    magnitude = magnitude / np.max(magnitude)

    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs, magnitude)
    plt.yscale("log")
    if harmonic_peaks is not None:
        for peak in harmonic_peaks:
            if peak < xlim:
                plt.axvline(
                    x=peak, color="r", linestyle="--", label=f"Peak: {peak:.2f} Hz"
                )
        if index == 1:
            plt.axvline(x=1645, color="g", linestyle="--", label=f"Peak: 1645 Hz")
        plt.legend(loc="upper right")

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (Normalized)")
    plt.xlim(0, xlim)
    plt.ylim(1e-6, 1.2)
    plt.grid(True)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)
    save_path = os.path.join(
        script_dir,
        "Overleaf",
        "data",
        "fft_spectrums",
        "fft_spectrum_segment_{}.png".format(index if index is not None else "unknown"),
    )
    print("Saving figure to:", save_path)
    plt.savefig(save_path)
    plt.show()
    plt.close()


def find_harmonic_peaks(
    fft_freqs,
    magnitude,
    num_peaks=4,
    peak_freq=None,
    threshold=0.001,
    rel_tol=0.05,
):
    mag = magnitude / np.max(magnitude)

    peaks = [peak_freq]

    for k in range(2, num_peaks + 1):
        target = k * peak_freq
        if target > fft_freqs[-1]:
            break
        tol = rel_tol * target
        idx = np.argmin(np.abs(fft_freqs - target))
        if abs(fft_freqs[idx] - target) <= tol and mag[idx] >= threshold:
            peaks.append(fft_freqs[idx])

    return peaks
