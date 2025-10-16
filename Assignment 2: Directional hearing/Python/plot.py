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
    plt.xlabel("Sample Index at 44.1 kHz")
    plt.ylabel("Amplitude")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_1/hrir_impulse_response_{angle}.png")
    plt.close()


def plot_itd_multiple(hrir_left, hrir_right, angles):
    os.makedirs("Overleaf/data/figures/task_1", exist_ok=True)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    colors = ["b", "r", "g", "m", "c", "y", "k"]
    offset = 0.15  # horizontal offset (in samples)
    x = np.arange(len(hrir_left[0]))

    for i in range(len(angles)):
        angle = angles[i]
        print(angles)
        color = colors[i % len(colors)]
        shift = (i - 1) * offset  # unique shift per angle

        stem_left = axs[0].stem(
            x + shift,
            hrir_left[i],
            linefmt=color + "-",
            markerfmt=color + "o",
            basefmt=" ",
            label=f"{angle}°",
        )
        stem_right = axs[1].stem(
            x + shift,
            hrir_right[i],
            linefmt=color + "-",
            markerfmt=color + "o",
            basefmt=" ",
            label=f"{angle}°",
        )

        # Set stem colors manually
        plt.setp(stem_left.markerline, color=color, alpha=0.7)
        plt.setp(stem_left.stemlines, color=color, alpha=0.7)
        plt.setp(stem_right.markerline, color=color, alpha=0.7)
        plt.setp(stem_right.stemlines, color=color, alpha=0.7)

    axs[0].set_title("Left Ear HRIRs")
    axs[1].set_title("Right Ear HRIRs")

    axs[1].set_xlabel("Sample Index at 44.1 kHz")
    axs[0].set_ylabel("Amplitude")
    axs[1].set_ylabel("Amplitude")

    for ax in axs:
        ax.grid(which="both", linestyle="--", linewidth=0.5)
        ax.legend(title="Angle (°)")

    fig.suptitle("HRIR Impulse Responses at Various Angles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(
        "Overleaf/data/figures/task_1/hrir_impulse_responses_multiple_angles.png"
    )
    plt.close(fig)


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


def plot_hrtf_response_multiple(f_vec, H_L_list, H_R_list, angles):
    os.makedirs("Overleaf/data/figures/task_2", exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    colors = ["b", "r", "g", "m", "c", "y", "k"]
    for i, angle in enumerate(angles):
        plt.semilogx(
            f_vec,
            20 * np.log10(np.abs(H_L_list[i])),
            label=f"Left Ear {angle}°",
            color=colors[i % len(colors)],
            linestyle="-",
        )
        plt.semilogx(
            f_vec,
            20 * np.log10(np.abs(H_R_list[i])),
            label=f"Right Ear {angle}°",
            color=colors[i % len(colors)],
            linestyle="--",
        )

    plt.title("HRTF Frequency Responses at Various Angles")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_vec[-1])
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig(
        "Overleaf/data/figures/task_2/hrtf_frequency_responses_multiple_angles.png"
    )
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
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_3/hrtfiir_freq_response_{angle}.png")
    plt.close()


def plot_hrtfiir_response_multiple(
    A1_list, B0_L_list, B1_L_list, B0_R_list, B1_R_list, f_s, angles
):
    os.makedirs("Overleaf/data/figures/task_3", exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    colors = ["b", "r", "g", "m", "c", "y", "k"]
    for i, angle in enumerate(angles):
        w, H_L = freqz([B0_L_list[i], B1_L_list[i]], [1, A1_list[i]], worN=512, fs=f_s)
        w, H_R = freqz([B0_R_list[i], B1_R_list[i]], [1, A1_list[i]], worN=512, fs=f_s)

        plt.semilogx(
            w,
            20 * np.log10(np.abs(H_L)),
            label=f"Left Ear {angle}°",
            color=colors[i % len(colors)],
        )
        plt.semilogx(
            w,
            20 * np.log10(np.abs(H_R)),
            label=f"Right Ear {angle}°",
            color=colors[i % len(colors)],
            linestyle="--",
        )

    plt.title("HRTF IIR Filter Frequency Responses at Various Angles")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_s / 2)
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig(
        "Overleaf/data/figures/task_3/hrtfiir_freq_responses_multiple_angles.png"
    )
    plt.close()


# ---------------------------------------------------------------
# Task 4: Combined HRIR + HRTF IIR (time and frequency domain)
# ---------------------------------------------------------------
def plot_combined_hrir_time(combined_left, combined_right, angle=45):
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


def plot_combined_hrir_freq(
    hrir_left, hrir_right, B0_L, B1_L, B0_R, B1_R, A1, f_s, angle
):
    os.makedirs("Overleaf/data/figures/task_4", exist_ok=True)

    # IIR frequency responses
    w, H_IIR_L = freqz([B0_L, B1_L], [1, A1], worN=2048, fs=f_s)
    w, H_IIR_R = freqz([B0_R, B1_R], [1, A1], worN=2048, fs=f_s)

    # HRIR (ITD) frequency responses (FIR)
    _, H_HRIR_L = freqz(hrir_left, [1], worN=2048, fs=f_s)
    _, H_HRIR_R = freqz(hrir_right, [1], worN=2048, fs=f_s)

    # Cascade = product
    H_tot_L = H_IIR_L * H_HRIR_L
    H_tot_R = H_IIR_R * H_HRIR_R

    plt.figure(figsize=(8, 4), dpi=300)
    plt.semilogx(w, 20 * np.log10(np.abs(H_tot_L)), label="Left Ear")
    plt.semilogx(w, 20 * np.log10(np.abs(H_tot_R)), label="Right Ear", linestyle="--")
    plt.title(f"Combined HRTF (IIR × ITD) at {angle}°")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_s / 2)
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig(f"Overleaf/data/figures/task_4/combined_hrir_freq_{angle}.png")
    plt.close()


def plot_combined_hrir_time_multiple(combined_left_list, combined_right_list, angles):
    os.makedirs("Overleaf/data/figures/task_4", exist_ok=True)

    # Time-domain plot
    plt.figure(figsize=(10, 6), dpi=300)

    colors = ["b", "r", "g", "m", "c", "y", "k"]
    for i, angle in enumerate(angles):
        plt.plot(
            combined_left_list[i],
            label=f"Left Ear {angle}°",
            color=colors[i % len(colors)],
        )
        plt.plot(
            combined_right_list[i],
            label=f"Right Ear {angle}°",
            color=colors[i % len(colors)],
            linestyle="--",
        )
    plt.title("Combined HRIR (HRIR + IIR) at Various Angles")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Overleaf/data/figures/task_4/combined_hrir_time_multiple.png")
    plt.close()


def plot_combined_hrir_freq_multiple(
    hrir_left_list,
    hrir_right_list,
    B0_L_list,
    B1_L_list,
    B0_R_list,
    B1_R_list,
    A1_list,
    f_s,
    angles,
):
    os.makedirs("Overleaf/data/figures/task_4", exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)

    colors = ["b", "r", "g", "m", "c", "y", "k"]
    for i, angle in enumerate(angles):
        # IIR frequency responses
        w, H_IIR_L = freqz(
            [B0_L_list[i], B1_L_list[i]], [1, A1_list[i]], worN=2048, fs=f_s
        )
        w, H_IIR_R = freqz(
            [B0_R_list[i], B1_R_list[i]], [1, A1_list[i]], worN=2048, fs=f_s
        )

        # HRIR (ITD) frequency responses (FIR)
        _, H_HRIR_L = freqz(hrir_left_list[i], [1], worN=2048, fs=f_s)
        _, H_HRIR_R = freqz(hrir_right_list[i], [1], worN=2048, fs=f_s)

        # Cascade = product
        H_tot_L = H_IIR_L * H_HRIR_L
        H_tot_R = H_IIR_R * H_HRIR_R

        plt.semilogx(
            w,
            20 * np.log10(np.abs(H_tot_L)),
            label=f"Left Ear {angle}°",
            color=colors[i % len(colors)],
        )
        plt.semilogx(
            w,
            20 * np.log10(np.abs(H_tot_R)),
            label=f"Right Ear {angle}°",
            color=colors[i % len(colors)],
            linestyle="--",
        )

    plt.title("Combined HRTF (IIR × ITD) at Various Angles")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.xlim(20, f_s / 2)
    plt.ylim(-20, 20)
    plt.tight_layout()
    plt.savefig("Overleaf/data/figures/task_4/combined_hrir_freq_multiple.png")
    plt.close()
