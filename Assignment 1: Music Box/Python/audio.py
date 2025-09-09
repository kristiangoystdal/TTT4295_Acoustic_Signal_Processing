import sounddevice as sd
import numpy as np
from time_splits import time_splits


def read_wav_file(wav_file):
    import numpy as np
    import wave

    # Load Wav file
    with wave.open(wav_file, "rb") as wf:
        sampwidth_bytes = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    # Decode to numpy
    if sampwidth_bytes == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16)
    elif sampwidth_bytes == 1:
        u8 = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.int16)
        audio = ((u8 - 128) << 8).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth_bytes*8} bits")

    return audio, framerate


def play_segment(segment, framerate):
    if segment.size != 0:
        sd.play(segment, samplerate=framerate, blocking=True)


def play_segments(segments, framerate):
    for i, segment in enumerate(segments):
        seg_start, seg_end = time_splits[i]
        print(f"Playing segment {i + 1}/{len(segments)}: {seg_start}s to {seg_end}s")
        play_segment(segment, framerate)
        if i < len(segments) - 1:
            input("Press Enter to play the next segment...")

    print("All segments played!")


def play_frequencies(peak_frequencies, duration=0.5):
    sample_rate = 44100
    for i, freq in enumerate(peak_frequencies):
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = 0.1 * np.sin(2 * np.pi * freq * t)
        sd.play(sine_wave, samplerate=sample_rate, blocking=True)
