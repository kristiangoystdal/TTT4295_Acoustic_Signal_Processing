import numpy as np
import math


# TASK 1: HRIR model
def hrir1(angle, head_radius, f_s, c):
    if angle < -90 or angle > 90:
        print("Angle must be between -90 and 90 degrees.")
        return None, None

    theta = math.radians(angle)
    delta_t = (head_radius / c) * (theta + math.sin(theta))
    n_delay = int(round(abs(delta_t) * f_s))

    # Max possible delay at 90° (~30 samples for default params)
    n_max = int(round((head_radius / c) * ((math.pi / 2) + 1) * f_s))

    hrir_left = np.zeros(n_max + 1)
    hrir_right = np.zeros(n_max + 1)

    if angle > 0:  # sound on right
        hrir_right[0] = 1.0
        hrir_left[n_delay] = 1.0
    elif angle < 0:  # sound on left
        hrir_left[0] = 1.0
        hrir_right[n_delay] = 1.0
    else:  # straight ahead
        hrir_left[0] = 1.0
        hrir_right[0] = 1.0

    return hrir_left, hrir_right
