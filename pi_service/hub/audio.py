"""WAV decoding using only the standard library and numpy (nothing extra to install)."""

import io
import wave

import numpy as np

TARGET_RATE = 16000  # YAMNet expects 16 kHz mono


def decode_wav(data):
    """Decode WAV bytes into (float32 samples in [-1, 1], sample_rate).

    Accepts 16-bit or 32-bit PCM, mono or stereo. Stereo is averaged to mono.
    """
    with wave.open(io.BytesIO(data), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())

    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width * 8} bits")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def to_16k(samples, rate):
    """Resample to 16 kHz with linear interpolation (fine for classification)."""
    if rate == TARGET_RATE or len(samples) == 0:
        return samples.astype(np.float32)
    duration = len(samples) / rate
    n_out = int(round(duration * TARGET_RATE))
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def load_clip(data):
    """WAV bytes -> float32 mono samples at 16 kHz."""
    samples, rate = decode_wav(data)
    return to_16k(samples, rate)


def encode_wav(samples, rate=TARGET_RATE):
    """float32 samples in [-1, 1] -> 16-bit mono WAV bytes (used by the simulator and tests)."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def rms_dbfs(samples):
    """Loudness of a clip in dBFS (0 = full scale, quieter is more negative)."""
    if len(samples) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-6))
