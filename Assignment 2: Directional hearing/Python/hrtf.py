# Author: Kristian Goystdal
# Date: 22.10.2025

import numpy as np
import math


# TASK 2: HRTF model
def hrtf1(angle, head_radius=0.09, f_s=44100, c=343, n_fft=512):
    theta = math.radians(angle)
    alpha_L = 1 + math.sin(theta)
    alpha_R = 1 - math.sin(theta)
    beta = 2 * c / head_radius

    f_vec = np.linspace(0, f_s / 2, n_fft // 2 + 1)
    omega = 2 * np.pi * f_vec

    num_L = np.sqrt((alpha_L * omega) ** 2 + beta**2)
    num_R = np.sqrt((alpha_R * omega) ** 2 + beta**2)
    denum = np.sqrt(omega**2 + beta**2)

    H_L = num_L / denum
    H_R = num_R / denum

    return f_vec, H_L, H_R


# TASK 3: HRTF IIR model
def hrtfiir(angle, head_radius=0.09, f_s=44100, c=343):
    T = 1 / f_s
    theta = math.radians(angle)
    alpha_L = 1 + math.sin(theta)
    alpha_R = 1 - math.sin(theta)
    beta = 2 * c / head_radius

    # Common
    a_0 = 2 + beta * T
    a_1 = beta * T - 2
    A_1 = a_1 / a_0

    # Left ear
    b_0_L = 2 + alpha_L + beta * T
    b_1_L = beta * T - 2 * alpha_L
    B_0_L = b_0_L / a_0
    B_1_L = b_1_L / a_0

    # Right ear
    b_0_R = 2 + alpha_R + beta * T
    b_1_R = beta * T - 2 * alpha_R
    B_0_R = b_0_R / a_0
    B_1_R = b_1_R / a_0

    return A_1, B_0_L, B_1_L, B_0_R, B_1_R
