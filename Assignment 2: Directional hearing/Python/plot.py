import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz


# ---------------------------------------------------------------
# Task 1: HRIR Impulse Response
# ---------------------------------------------------------------
def plot_itd(hrir_left, hrir_right, angle=90):
    os.makedirs("Overleaf/data/figures/task_1", exist_ok=True)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.stem(hrir_left, linefmt="b-", markerfmt="bo", basefmt=" ", label="Left Ear")
    plt.stem(hrir_right, linefmt="r-", markerfmt="ro", basefmt=" ", label="Right Ear")
    plt.title(f"HRIR Impulse Response at {angle}°")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_1/hrir_impulse_response_{angle}.png")
    plt.close()


# ---------------------------------------------------------------
# Task 2: Analytical HRTF Frequency Response
# ---------------------------------------------------------------
def plot_hrtf_response(f_vec, H_L, H_R, angle=90):
    os.makedirs("Overleaf/data/figures/task_2", exist_ok=True)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogx(f_vec, 20 * np.log10(np.abs(H_L)), label="Left Ear", color="blue")
    plt.semilogx(
        f_vec,
        20 * np.log10(np.abs(H_R)),
        label="Right Ear",
        color="red",
        linestyle="--",
    )
    plt.title(f"HRTF Frequency Response at {angle}°")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_vec[-1])
    plt.ylim(-30, 10)
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_2/hrtf_frequency_response_{angle}.png")
    plt.close()


# ---------------------------------------------------------------
# Task 3: HRTF IIR Filter Frequency Response
# ---------------------------------------------------------------
def plot_hrtfiir_response(A1, B0_L, B1_L, B0_R, B1_R, f_s, angle=90):
    os.makedirs("Overleaf/data/figures/task_3", exist_ok=True)

    w, H_L = freqz([B0_L, B1_L], [1, A1], worN=512, fs=f_s)
    w, H_R = freqz([B0_R, B1_R], [1, A1], worN=512, fs=f_s)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogx(w, 20 * np.log10(np.abs(H_L)), label="Left Ear", color="blue")
    plt.semilogx(
        w, 20 * np.log10(np.abs(H_R)), label="Right Ear", color="red", linestyle="--"
    )
    plt.title(f"HRTF IIR Filter Frequency Response at {angle}°")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_s / 2)
    plt.ylim(-30, 10)
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_3/hrtfiir_freq_response_{angle}.png")
    plt.close()


# ---------------------------------------------------------------
# Task 4: Combined HRIR + HRTF IIR (time and frequency domain)
# ---------------------------------------------------------------
def plot_combined_hrir(combined_left, combined_right, angle=45):
    os.makedirs("Overleaf/data/figures/task_4", exist_ok=True)

    # Time-domain plot
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(combined_left, label="Left Ear", color="blue")
    plt.plot(combined_right, label="Right Ear", color="red", linestyle="--")
    plt.title(f"Combined HRIR (HRIR + IIR) at {angle}°")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_4/combined_hrir_time_{angle}.png")
    plt.close()

    # Frequency-domain plot
    w, H_L = freqz(combined_left, worN=512, fs=44100)
    w, H_R = freqz(combined_right, worN=512, fs=44100)
    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogx(w, 20 * np.log10(np.abs(H_L)), label="Left Ear", color="blue")
    plt.semilogx(
        w, 20 * np.log10(np.abs(H_R)), label="Right Ear", color="red", linestyle="--"
    )
    plt.title(f"Combined HRIR Frequency Response at {angle}°")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, 44100 / 2)
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_4/combined_hrir_freq_{angle}.png")
    plt.close()
