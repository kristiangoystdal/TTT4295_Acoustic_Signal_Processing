import sounddevice as sd
import numpy as np
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
