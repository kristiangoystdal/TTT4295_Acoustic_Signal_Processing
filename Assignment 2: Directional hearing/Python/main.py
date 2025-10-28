# Author: Kristian Goystdal
# Date: 22.10.2025

import numpy as np
import sounddevice as sd
from scipy.signal import lfilter

from plot import *
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

    signal_constant = np.ones(512)

    filtered_left = lfilter(
        [hrtfiir_left_0, hrtfiir_left_1], [1, hrtfiir_a], signal_constant
    )
    filtered_right = lfilter(
        [hrtfiir_right_0, hrtfiir_right_1], [1, hrtfiir_a], signal_constant
    )

    combined_left = np.convolve(filtered_left, hrir_left, "full")
    combined_right = np.convolve(filtered_right, hrir_right, "full")

    peak = max(np.max(np.abs(combined_left)), np.max(np.abs(combined_right)))
    combined_left /= peak
    combined_right /= peak

    plot_combined_hrir_time(combined_left, combined_right, angle)
    plot_combined_hrir_freq(
        hrir_left,
        hrir_right,
        hrtfiir_left_0,
        hrtfiir_left_1,
        hrtfiir_right_0,
        hrtfiir_right_1,
        hrtfiir_a,
        f_s,
        angle,
    )


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

    filtered_left = lfilter(
        [hrtfiir_left_0, hrtfiir_left_1], [1, hrtfiir_a], signal_constant
    )
    filtered_right = lfilter(
        [hrtfiir_right_0, hrtfiir_right_1], [1, hrtfiir_a], signal_constant
    )

    combined_left = np.convolve(filtered_left, hrir_left, "full")
    combined_right = np.convolve(filtered_right, hrir_right, "full")

    # Combine all the segments and play sound
    duration = 10  # seconds
    delay = 5  # seconds
    t = np.linspace(0, duration, int(f_s * duration), endpoint=False)

    mono_signal = 0.5 * pink_noise(len(t))
    stereo_signal = np.zeros((len(mono_signal), 2))
    stereo_signal[:, 0] = np.convolve(mono_signal, combined_left, "same")
    stereo_signal[:, 1] = np.convolve(mono_signal, combined_right, "same")
    if full_stereo_signal is None:
        full_stereo_signal = stereo_signal
    else:
        full_stereo_signal = np.vstack((full_stereo_signal, stereo_signal))

    # Add delay between angles
    delay_samples = int(f_s * delay)
    silence = np.zeros((delay_samples, 2))
    full_stereo_signal = np.vstack((full_stereo_signal, silence))

full_stereo_signal /= np.max(np.abs(full_stereo_signal))

plot_sound_demo(full_stereo_signal, f_s)

# sd.play(full_stereo_signal, f_s)
# sd.wait()


# Plot 0, 30 and 90 degrees together
(
    hrir_left_list,
    hrir_right_list,
    f_vec_list,
    H_L_list,
    H_R_list,
    hrtfiir_a_list,
    hrtfiir_left_0_list,
    hrtfiir_left_1_list,
    hrtfiir_right_0_list,
    hrtfiir_right_1_list,
    combined_left_list,
    combined_right_list,
) = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
angles = [-30, 0, 90]
for angle in angles:
    # Task 1
    hrir_left, hrir_right = hrir1(angle, head_radius, f_s, c)

    # Task 2
    f_vec, H_L, H_R = hrtf1(angle, head_radius, f_s, c)

    # Task 3
    hrtfiir_a, hrtfiir_left_0, hrtfiir_left_1, hrtfiir_right_0, hrtfiir_right_1 = (
        hrtfiir(angle, head_radius, f_s, c)
    )
    # Task 4
    signal_constant = np.ones(512)

    filtered_left = lfilter(
        [hrtfiir_left_0, hrtfiir_left_1], [1, hrtfiir_a], signal_constant
    )
    filtered_right = lfilter(
        [hrtfiir_right_0, hrtfiir_right_1], [1, hrtfiir_a], signal_constant
    )

    combined_left = np.convolve(filtered_left, hrir_left, "full")
    combined_right = np.convolve(filtered_right, hrir_right, "full")

    peak = max(np.max(np.abs(combined_left)), np.max(np.abs(combined_right)))
    combined_left /= peak
    combined_right /= peak

    # Append to lists
    hrir_left_list.append(hrir_left)
    hrir_right_list.append(hrir_right)
    f_vec_list.append(f_vec)
    H_L_list.append(H_L)
    H_R_list.append(H_R)
    hrtfiir_a_list.append(hrtfiir_a)
    hrtfiir_left_0_list.append(hrtfiir_left_0)
    hrtfiir_left_1_list.append(hrtfiir_left_1)
    hrtfiir_right_0_list.append(hrtfiir_right_0)
    hrtfiir_right_1_list.append(hrtfiir_right_1)
    combined_left_list.append(combined_left)
    combined_right_list.append(combined_right)

plot_itd_multiple(hrir_left_list, hrir_right_list, angles)
plot_hrtf_response_multiple(f_vec_list, H_L_list, H_R_list, angles)
plot_hrtfiir_response_multiple(
    hrtfiir_a_list,
    hrtfiir_left_0_list,
    hrtfiir_left_1_list,
    hrtfiir_right_0_list,
    hrtfiir_right_1_list,
    f_s,
    angles,
)
plot_combined_hrir_time_multiple(combined_left_list, combined_right_list, angles)
plot_combined_hrir_freq_multiple(
    hrir_left_list,
    hrir_right_list,
    hrtfiir_left_0_list,
    hrtfiir_left_1_list,
    hrtfiir_right_0_list,
    hrtfiir_right_1_list,
    hrtfiir_a_list,
    f_s,
    angles,
)
