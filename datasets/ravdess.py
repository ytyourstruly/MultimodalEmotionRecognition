# -*- coding: utf-8 -*-
import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa

DEFAULT_SR      = 22050
DEFAULT_N_MFCC  = 40
DEFAULT_N_MELS  = 128
DEFAULT_T_FIXED = 128
DEFAULT_FMAX    = 8000

# ── Feature group layout in the 255-channel stack ────────────────────────────
# get_audio_features stacks: [mfcc_block, log_mel, contrast]
#   mfcc_block : n_mfcc * 3 = 120  → indices   0 : 120
#   log_mel    : n_mels     = 128  → indices 120 : 248
#   contrast   : 7 bands           → indices 248 : 255
_FEATURE_GROUPS = {
    'mfcc':     (0,   120),   # 120 channels  (40 base + 40 Δ + 40 ΔΔ)
    'log_mel':  (120, 248),   # 128 channels
    'contrast': (248, 255),   #   7 channels
}
_TOTAL_CHANNELS = 255


def get_mfccs(y, sr, audio_channels=None):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=audio_channels or DEFAULT_N_MFCC)
    return mfcc
def get_proportional_indices(n_channels: int) -> np.ndarray:
    """
    Return n_channels indices drawn equally from each feature group.

    Each group contributes floor(n_channels / n_groups) channels,
    with any remainder distributed one-at-a-time to the largest groups first.
    Indices are sampled uniformly within each group and returned sorted
    so channel order (spatial coherence) is preserved.

    Examples
    --------
    n_channels=9  → 3 from mfcc, 3 from log_mel, 3 from contrast
    n_channels=10 → 4 from mfcc, 3 from log_mel, 3 from contrast
    n_channels=255 → all channels (identity, no masking benefit)
    n_channels=7  → 3 from mfcc, 3 from log_mel, 1 from contrast
                    (contrast only has 7 total; evenly sampled within it)
    """
    if n_channels <= 0:
        raise ValueError(f'n_channels must be > 0, got {n_channels}')
    if n_channels > _TOTAL_CHANNELS:
        raise ValueError(
            f'n_channels ({n_channels}) exceeds total channels ({_TOTAL_CHANNELS})'
        )

    groups      = list(_FEATURE_GROUPS.items())
    n_groups    = len(groups)
    base        = n_channels // n_groups
    remainder   = n_channels %  n_groups

    # Sort by group size descending so remainder goes to richest groups first
    groups_sorted = sorted(groups, key=lambda kv: kv[1][1] - kv[1][0], reverse=True)

    selected = []
    for rank, (name, (start, end)) in enumerate(groups_sorted):
        group_size = end - start
        n_take     = base + (1 if rank < remainder else 0)
        n_take     = min(n_take, group_size)   # cap at group capacity

        # Evenly-spaced (linspace) indices within the group — more
        # representative than random when n_take is small
        local_idx  = np.round(
            np.linspace(0, group_size - 1, n_take)
        ).astype(int)
        selected.append(start + local_idx)

    indices = np.sort(np.concatenate(selected))
    assert len(indices) == n_channels or n_channels > _TOTAL_CHANNELS, \
        f'Expected {n_channels} indices, got {len(indices)}'
    return indices


def _pad_or_truncate(arr: np.ndarray, t_fixed: int) -> np.ndarray:
    t = arr.shape[-1]
    if t == t_fixed:
        return arr
    if t > t_fixed:
        start = (t - t_fixed) // 2
        return arr[..., start:start + t_fixed]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, t_fixed - t)]
    return np.pad(arr, pad_width, mode='constant')


def _zscore(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = arr.mean(axis=-1, keepdims=True)
    std  = arr.std(axis=-1,  keepdims=True) + eps
    return (arr - mean) / std


def get_audio_features(
    y, sr=DEFAULT_SR, n_mfcc=DEFAULT_N_MFCC, n_mels=DEFAULT_N_MELS,
    fmax=DEFAULT_FMAX, t_fixed=DEFAULT_T_FIXED, normalise=True, add_deltas=True,
) -> np.ndarray:
    """Returns (255, T_fixed) float32 always — Fisher/proportional masking applied outside."""
    stft = np.abs(librosa.stft(y))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_block = np.vstack([
        mfccs,
        librosa.feature.delta(mfccs, order=1),
        librosa.feature.delta(mfccs, order=2),
    ]) if add_deltas else mfccs

    log_mel  = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax),
        ref=np.max,
    )
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)

    aligned = [_pad_or_truncate(f, t_fixed) for f in [mfcc_block, log_mel, contrast]]
    if normalise:
        aligned = [_zscore(f) for f in aligned]
    return np.vstack(aligned).astype(np.float32)


def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    return [Image.fromarray(video[i]) for i in range(video.shape[0])]


def get_default_video_loader():
    return functools.partial(video_loader)


def load_audio(audiofile, sr):
    y, sr = librosa.core.load(audiofile, sr=sr)
    return y, sr


def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()

    dataset = []
    seen_subsets = set()                           # ← track what IS in the file

    for line in annots:
        line = line.strip()
        if not line:
            continue
        parts = line.split(';')
        if len(parts) != 4:
            continue
        filename, audiofilename, label, trainvaltest = parts
        seen_subsets.add(trainvaltest.strip())
        if trainvaltest.strip() != subset:
            continue
        dataset.append({
            'video_path': filename,
            'audio_path': audiofilename,
            'label':      int(label) - 1,
        })

    # ── Guard: empty dataset → clear diagnostic instead of cryptic DataLoader crash
    if len(dataset) == 0:
        raise ValueError(
            f"\n[RAVDESS] make_dataset returned 0 samples for subset='{subset}'.\n"
            f"  Annotation file : {annotation_path}\n"
            f"  Subsets found   : {sorted(seen_subsets)}\n"
            f"  Expected one of : 'train', 'val', 'test'\n"
            f"  → Check that the 4th column in your annotation file "
            f"matches the subset string exactly (case-sensitive, no trailing spaces)."
        )

    return dataset

class RAVDESS(data.Dataset):
    def __init__(self,                 
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader,
                 data_type='audiovisual',
                 audio_transform=None,
                 fisher_indices_path: str = None,
                 audio_channels: int = None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type
        self.audio_channels = audio_channels

        assert data_type in ('audio', 'video', 'audiovisual'), \
            f'Unknown data_type: {data_type}'  # ← catches misconfiguration early

        if fisher_indices_path is not None:
            indices = np.load(fisher_indices_path)
            self.fisher_indices = np.sort(indices)
            print(f"[RAVDESS] Fisher mask loaded: "
                  f"{len(self.fisher_indices)} / 255 channels kept "
                  f"({fisher_indices_path})")
        else:
            self.fisher_indices = None

    def _load_audio_features(self, path) -> np.ndarray:
        """
        Always returns a fixed-shape (C, T_fixed) array.
        
        Bug fixed: original code used get_mfccs() when fisher_indices=None,
        which returns (10, T) with variable T — DataLoader cannot collate
        variable-length arrays into a batch. Now always uses get_audio_features()
        with T_fixed=128 regardless of Fisher masking.
        """
        y, sr = load_audio(path, sr=DEFAULT_SR)

        if self.audio_transform is not None:
            self.audio_transform.randomize_parameters()
            y = self.audio_transform(y)

        # Always use full feature extraction for fixed shape
        features = get_mfccs(y, sr, audio_channels=self.audio_channels)           # (255, T_fixed)

        if self.fisher_indices is not None:
            features = features[self.fisher_indices, :]  # (k, T_fixed)

        return features.astype(np.float32)

    def __getitem__(self, index):
        target = self.data[index]['label']

        # ── Video ────────────────────────────────────────────────────
        if self.data_type in ('video', 'audiovisual'):
            path = self.data[index]['video_path']
            clip = self.loader(path)
            if self.spatial_transform is not None:
                self.spatial_transform.randomize_parameters()
                clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

            if self.data_type == 'video':
                return clip, target

        # ── Audio ────────────────────────────────────────────────────
        if self.data_type in ('audio', 'audiovisual'):
            audio_features = self._load_audio_features(
                self.data[index]['audio_path']
            )
            if self.data_type == 'audio':
                return audio_features, target

        # ── Audiovisual ───────────────────────────────────────────────
        return audio_features, clip, target

    def __len__(self):
        return len(self.data)