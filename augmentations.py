import torch
import numpy as np
import librosa
import random
from scipy.signal import butter, lfilter


# ─── Primitives ───────────────────────────────────────────────────────────────

class NormalizeAudio:
    """RMS normalization. Apply LAST."""
    def __call__(self, y: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(y ** 2)) + 1e-9
        return y / rms

    def randomize_parameters(self): pass


class RandomGainVariation:
    """
    Tight gain variation: preserves relative energy as an emotional cue.
    Use min_gain=0.85, max_gain=1.15 during fusion (not 0.7–1.3).
    """
    def __init__(self, min_gain=0.85, max_gain=1.15, p=0.4):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p
        self.gain = 1.0

    def randomize_parameters(self):
        self.gain = random.uniform(self.min_gain, self.max_gain)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return y * self.gain if self.do_apply else y


class RandomNoiseInjection:
    """
    Broader SNR range than original (10–35 dB) for real-world robustness.
    """
    def __init__(self, min_snr_db=10, max_snr_db=35, p=0.5):
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.p = p
        self.snr_db = 30

    def randomize_parameters(self):
        self.snr_db = random.uniform(self.min_snr_db, self.max_snr_db)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if not self.do_apply:
            return y
        signal_power = np.mean(y ** 2) + 1e-9
        noise_power  = signal_power / (10 ** (self.snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return (y + noise).astype(np.float32)


class RandomPitchShift:
    """
    Pitch shift WITHOUT duration change (safe for multimodal sync).
    ±2 semitones during fusion, ±3 during audio-only pretraining.
    
    Emotion boundary note:
      >±3 semitones risks perceptual emotion class shift.
      ±1-2 semitones = speaker normalization without label corruption.
    """
    def __init__(self, sr=22050, min_steps=-2, max_steps=2, p=0.5):
        self.sr = sr
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.p = p
        self.n_steps = 0

    def randomize_parameters(self):
        self.n_steps = random.uniform(self.min_steps, self.max_steps)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if not self.do_apply or abs(self.n_steps) < 0.1:
            return y
        return librosa.effects.pitch_shift(
            y, sr=self.sr, n_steps=self.n_steps
        ).astype(np.float32)


class RandomTimeStretch:
    """
    WARNING: Changes audio duration → BREAKS audio-video sync.
    Use ONLY during audio-only pretraining, NOT during multimodal fusion.
    
    Rate < 1.0 = slower (lower arousal percept)
    Rate > 1.0 = faster (higher arousal percept)
    Keep within 0.85–1.15 to avoid emotion label corruption.
    """
    def __init__(self, min_rate=0.85, max_rate=1.15, p=0.4):
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.p = p
        self.rate = 1.0

    def randomize_parameters(self):
        self.rate = random.uniform(self.min_rate, self.max_rate)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if not self.do_apply or abs(self.rate - 1.0) < 0.01:
            return y
        stretched = librosa.effects.time_stretch(y, rate=self.rate)
        # Re-align to original length (pad/truncate) so downstream
        # get_audio_features receives a fixed-size waveform
        target_len = len(y)
        if len(stretched) > target_len:
            start = (len(stretched) - target_len) // 2
            return stretched[start:start + target_len].astype(np.float32)
        pad = target_len - len(stretched)
        return np.pad(stretched, (0, pad), mode='constant').astype(np.float32)


class RandomReverberation:
    """
    Synthetic room impulse response via exponential decay model.
    Safe for multimodal fusion — does not shift timing.
    RT60 controls perceived room size: 0.1s (small room) – 0.8s (hall).
    """
    def __init__(self, sr=22050, min_rt60=0.05, max_rt60=0.6, p=0.4):
        self.sr = sr
        self.min_rt60 = min_rt60
        self.max_rt60 = max_rt60
        self.p = p

    def randomize_parameters(self):
        self.rt60 = random.uniform(self.min_rt60, self.max_rt60)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if not self.do_apply:
            return y
        # Exponential decay IR (fast synthetic approximation)
        ir_len = int(self.rt60 * self.sr)
        ir = np.exp(-3 * np.linspace(0, self.rt60, ir_len) / self.rt60)
        ir = ir * np.random.randn(ir_len)  # stochastic IR
        ir = ir / (np.abs(ir).max() + 1e-9)
        reverbed = np.convolve(y, ir, mode='full')[:len(y)]
        # Mix dry/wet to preserve intelligibility
        wet_ratio = random.uniform(0.1, 0.4)
        return ((1 - wet_ratio) * y + wet_ratio * reverbed).astype(np.float32)


class RandomTelephoneEffect:
    """
    Simulates telephone/codec bandwidth (300–3400 Hz bandpass).
    Safe for multimodal fusion.
    Useful for: RAVDESS (studio) → real-world deployment gap.
    Does NOT significantly affect F0 or energy envelope for emotion.
    """
    def __init__(self, sr=22050, p=0.25):
        self.sr = sr
        self.p = p

    def randomize_parameters(self):
        self.do_apply = random.random() < self.p

    def _bandpass(self, y, lowcut=300, highcut=3400, order=4):
        nyq = self.sr / 2.0
        b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, y).astype(np.float32)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self._bandpass(y) if self.do_apply else y


class RandomTimeShift:
    """
    WARNING: Shifts waveform in time → BREAKS audio-video sync if video
    frames are not shifted by the same amount.
    Use ONLY during audio-only pretraining OR if your DataLoader
    applies the same shift index to video frames.
    """
    def __init__(self, max_shift_ratio=0.1, p=0.5):
        self.max_shift_ratio = max_shift_ratio
        self.p = p

    def randomize_parameters(self):
        self.shift_ratio = random.uniform(-self.max_shift_ratio,
                                          self.max_shift_ratio)
        self.do_apply = random.random() < self.p

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if not self.do_apply:
            return y
        shift = int(self.shift_ratio * len(y))
        return np.roll(y, shift).astype(np.float32)


# ─── Composed Pipelines ───────────────────────────────────────────────────────

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

    def __call__(self, y):
        for t in self.transforms:
            y = t(y)
        return y


def get_audio_transform_pretrain(sr=22050, max_shift_ratio=0.1):
    """
    Audio-ONLY pretraining pipeline.
    Stronger augmentation: includes time stretch and time shift.
    These break A/V sync so must not be used during multimodal fusion.
    """
    return Compose([
        RandomGainVariation(min_gain=0.80, max_gain=1.20, p=0.5),
        RandomPitchShift(sr=sr, min_steps=-3, max_steps=3, p=0.5),
        RandomTimeStretch(min_rate=0.85, max_rate=1.15, p=0.4),   # pretrain only
        RandomNoiseInjection(min_snr_db=10, max_snr_db=35, p=0.5),
        RandomReverberation(sr=sr, min_rt60=0.05, max_rt60=0.7, p=0.4),
        RandomTelephoneEffect(sr=sr, p=0.25),
        RandomTimeShift(max_shift_ratio=max_shift_ratio, p=0.4),   # pretrain only
        NormalizeAudio(),   # always last
    ])


def get_audio_transform_fusion(sr=22050):
    """
    Multimodal fusion pipeline.
    Sync-safe only: NO time stretch, NO time shift.
    Tighter pitch range, tighter gain variation.
    """
    return Compose([
        RandomGainVariation(min_gain=0.85, max_gain=1.15, p=0.4),
        RandomPitchShift(sr=sr, min_steps=-2, max_steps=2, p=0.4),
        # NO RandomTimeStretch — breaks A/V sync
        RandomNoiseInjection(min_snr_db=15, max_snr_db=35, p=0.4),
        RandomReverberation(sr=sr, min_rt60=0.05, max_rt60=0.5, p=0.35),
        RandomTelephoneEffect(sr=sr, p=0.2),
        # NO RandomTimeShift — breaks A/V sync
        NormalizeAudio(),   # always last
    ])


def get_audio_transform_val():
    """Validation/test: normalization only."""
    return Compose([NormalizeAudio()])