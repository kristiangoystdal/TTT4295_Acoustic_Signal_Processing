import numpy as np
import sounddevice as sd
from scipy.signal import lfilter

from plot import (
    plot_itd,
    plot_hrtf_response,
    plot_hrtfiir_response,
    plot_combined_hrir,
)
from hrir import hrir1
from hrtf import hrtf1, hrtfiir


# Constants
f_s = 44100  # Hz
c = 343  # m/s
head_radius = 0.09  # meters


# Task 1-4: HRTF and HRIR calculations
for angle in [-90, -60, -30, 0, 30, 60, 90]:
    print(f"Calculating for angle: {angle} degrees")

    # Task 1: HRIR
    hrir_left, hrir_right = hrir1(angle, head_radius, f_s, c)
    plot_itd(hrir_left, hrir_right, angle)

    # Task 2: HRTF
    f_vec, H_L, H_R = hrtf1(angle, head_radius, f_s, c)
    plot_hrtf_response(f_vec, H_L, H_R, angle)

    # Task 3: HRTF IIR
    hrtfiir_a, hrtfiir_left_0, hrtfiir_left_1, hrtfiir_right_0, hrtfiir_right_1 = (
        hrtfiir(angle, head_radius, f_s, c)
    )
    plot_hrtfiir_response(
        hrtfiir_a,
        hrtfiir_left_0,
        hrtfiir_left_1,
        hrtfiir_right_0,
        hrtfiir_right_1,
        f_s,
        angle,
    )

    # Task 4: Combined HRIR and HRTF IIR

    # combined_left = np.convolve(hrir_left, [hrtfiir_left_0, hrtfiir_left_1], "full")
    # combined_right = np.convolve(hrir_right, [hrtfiir_right_0, hrtfiir_right_1], "full")

    combined_left = lfilter([hrtfiir_left_0, hrtfiir_left_1], [1, hrtfiir_a], hrir_left)
    combined_right = lfilter(
        [hrtfiir_right_0, hrtfiir_right_1], [1, hrtfiir_a], hrir_right
    )

    plot_combined_hrir(combined_left, combined_right, angle)


# Generate pink noise
def pink_noise(N):
    n_rows = 16
    n_cols = N
    array = np.random.randn(n_rows, n_cols)
    array = np.cumsum(array, axis=1)
    pink = np.sum(array, axis=0)
    pink /= np.max(np.abs(pink))
    return pink


# Task 5: Sound playback
step = 30
full_stereo_signal = None
for angle in range(-90, 91, step):
    print(f"Playing sound at {angle} degrees")
    hrir_left, hrir_right = hrir1(angle, head_radius, f_s, c)

    hrtfiir_a, hrtfiir_left_0, hrtfiir_left_1, hrtfiir_right_0, hrtfiir_right_1 = (
        hrtfiir(angle, head_radius, f_s, c)
    )

    combined_left = np.convolve(hrir_left, [hrtfiir_left_0, hrtfiir_left_1], "full")
    combined_right = np.convolve(hrir_right, [hrtfiir_right_0, hrtfiir_right_1], "full")

    # Combine all the segments and play sound
    duration = 1  # seconds
    t = np.linspace(0, duration, int(f_s * duration), endpoint=False)

    mono_signal = 0.5 * pink_noise(len(t))
    stereo_signal = np.zeros((len(mono_signal), 2))
    stereo_signal[:, 0] = np.convolve(mono_signal, combined_left, "same")
    stereo_signal[:, 1] = np.convolve(mono_signal, combined_right, "same")
    if full_stereo_signal is None:
        full_stereo_signal = stereo_signal
    else:
        full_stereo_signal = np.vstack((full_stereo_signal, stereo_signal))

sd.play(full_stereo_signal, f_s)
sd.wait()
