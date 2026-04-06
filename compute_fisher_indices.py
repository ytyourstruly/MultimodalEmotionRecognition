"""
compute_fisher_indices.py
─────────────────────────
Pre-training script: compute Fisher scores over the TRAINING split of every
fold annotation file, saving the selected channel indices under a per-fold
sub-directory.  Run once before main.py.

Usage
-----
    python compute_fisher_indices.py \
        --annotation_paths \
            /data/annotations/annotations_croppad_fold1.txt \
            /data/annotations/annotations_croppad_fold2.txt \
        --fisher_threshold 0.95 \
        --output_dir /results/

Output layout
─────────────
    <output_dir>/
        fold_1/fisher_indices.npy
        fold_2/fisher_indices.npy

Each .npy is a 1-D int64 array of selected indices, sorted channel-order.
The number of indices is determined automatically by the cumulative energy
threshold — no manual k required.

Fisher score (per channel c):
    F(c) = Σ_k n_k (μ_kc − μ_c)²  /  Σ_k n_k σ²_kc
"""

import argparse
import os
import numpy as np
from tqdm import tqdm

# ── feature blocks ────────────────────────────────────────────────────────────

BLOCKS = [
    ('MFCC+Δ+ΔΔ  (0–119)',   0,   120),
    ('LogMel     (120–247)', 120,  248),
    ('Contrast   (248–254)', 248,  255),
]
N_CHANNELS = 255


# ── Fisher score (pure-numpy fallback) ───────────────────────────────────────

def fisher_score_numpy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return per-channel Fisher discriminant score. X: (N,C), y: (N,)."""
    mu_all      = X.mean(axis=0)
    numerator   = np.zeros(X.shape[1])
    denominator = np.zeros(X.shape[1])

    for cls in np.unique(y):
        X_k   = X[y == cls]
        n_k   = len(X_k)
        mu_k  = X_k.mean(axis=0)
        numerator   += n_k * (mu_k - mu_all) ** 2
        denominator += n_k * X_k.var(axis=0)

    return numerator / np.maximum(denominator, 1e-10)


# ── automatic channel selection by cumulative energy ─────────────────────────

def select_by_cumulative_energy(scores: np.ndarray,
                                threshold: float = 0.95) -> np.ndarray:
    """
    Return the fewest top channels whose cumulative Fisher score mass
    reaches `threshold` of the total.  Result is sorted in channel order.
    """
    sorted_idx = np.argsort(scores)[::-1]
    cumulative = np.cumsum(scores[sorted_idx])
    k = int(np.searchsorted(cumulative, threshold * cumulative[-1])) + 1
    print(f"[Fisher] Threshold {threshold:.0%} → {k} / {len(scores)} channels selected")
    return np.sort(sorted_idx[:k])


# ── feature extraction ────────────────────────────────────────────────────────

def extract_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Return (C, T) float32 feature matrix — time axis preserved.
    C = 255:  120 MFCC+Δ+ΔΔ  |  128 Log-Mel  |  7 Spectral Contrast
    """
    import soundfile as sf
    import librosa

    data, sr_in = sf.read(audio_path, dtype='float32', always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_in != sr:
        n_out = int(len(data) * sr / sr_in)
        data  = np.interp(
            np.linspace(0, 1, n_out, endpoint=False),
            np.linspace(0, 1, len(data), endpoint=False),
            data,
        ).astype(np.float32)

    stft  = np.abs(librosa.stft(data))
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

    return np.vstack([
        mfccs,
        librosa.feature.delta(mfccs, order=1),
        librosa.feature.delta(mfccs, order=2),                     # (120, T)
        librosa.power_to_db(
            librosa.feature.melspectrogram(y=data, sr=sr,
                                           n_mels=128, fmax=8000),
            ref=np.max),                                            # (128, T)
        librosa.feature.spectral_contrast(S=stft, sr=sr),          # (  7, T)
    ]).astype(np.float32)                                           # (255, T)


# ── annotation parser ─────────────────────────────────────────────────────────

def load_train_paths_labels(annotation_path: str):
    paths, labels = [], []
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 4:
                continue
            _, audio_path, label, split = parts
            if split.strip() == 'training':
                paths.append(audio_path.strip())
                labels.append(int(label) - 1)
    return paths, labels


# ── per-fold processing ───────────────────────────────────────────────────────

def process_fold(annotation_path: str, out_path: str,
                 threshold: float, sr: int, use_skfeature: bool) -> None:

    print(f"\n[Fisher] Annotation : {annotation_path}")
    paths, labels = load_train_paths_labels(annotation_path)
    print(f"[Fisher] Training samples: {len(paths)}")

    # extract and mean-pool over time (for Fisher scoring only)
    X_list, failed = [], 0
    for path in tqdm(paths, desc='Extracting', leave=True):
        try:
            X_list.append(extract_features(path, sr=sr).mean(axis=1))
        except Exception as exc:
            print(f"  WARNING: skipping {path}: {exc}")
            failed += 1

    if failed:
        print(f"[Fisher] {failed} file(s) skipped")
    if not X_list:
        print("[Fisher] ERROR: no valid samples — skipping fold.")
        return

    X = np.stack(X_list)                          # (N, 255)
    y = np.array(labels[:len(X_list)])
    print(f"[Fisher] Feature matrix: {X.shape}")

    # standardise before scoring
    X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)

    # compute scores
    if use_skfeature:
        from skfeature.function.similarity_based import fisher_score as fs_lib
        scores = fs_lib.fisher_score(X_scaled, y)
        print("[Fisher] Using skfeature")
    else:
        scores = fisher_score_numpy(X_scaled, y)
        print("[Fisher] Using numpy fallback")

    # automatic selection
    fs_indices = select_by_cumulative_energy(scores, threshold)

    # diagnostics
    print(f"[Fisher] Top-10 indices : {fs_indices[:10]}")
    print(f"[Fisher] Score range    : min={scores.min():.4f}  max={scores.max():.4f}")
    print("\n[Fisher] Block breakdown:")
    for name, lo, hi in BLOCKS:
        count = ((fs_indices >= lo) & (fs_indices < hi)).sum()
        print(f"  {name:30s}: {count:3d} / {hi - lo}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, fs_indices.astype(np.int64))
    print(f"\n[Fisher] Saved {len(fs_indices)} indices → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Fisher feature-selection indices for one or more folds."
    )
    parser.add_argument('--annotation_paths', nargs='+', required=True,
                        metavar='FILE',
                        help='Fold annotation .txt files (fold index = list order)')
    parser.add_argument('--fisher_threshold', type=float, default=0.95,
                        metavar='FLOAT',
                        help='Cumulative energy fraction to retain (default: 0.95)')
    parser.add_argument('--output_dir', required=True, metavar='DIR',
                        help='Root output dir; fold n → <dir>/fold_n/fisher_indices.npy')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Audio sample rate (default: 22050)')
    args = parser.parse_args()

    try:
        from skfeature.function.similarity_based import fisher_score as _  # noqa
        use_skfeature = True
    except ImportError:
        use_skfeature = False

    n_folds = len(args.annotation_paths)
    print(f"\n[Fisher] {n_folds} fold(s) | threshold={args.fisher_threshold} | root={args.output_dir}")

    for fold_idx, annotation_path in enumerate(args.annotation_paths, start=1):
        print(f"\n{'═' * 60}")
        print(f"[Fisher] Fold {fold_idx}/{n_folds}")
        print(f"{'═' * 60}")
        process_fold(
            annotation_path = annotation_path,
            out_path        = os.path.join(args.output_dir, f"fold_{fold_idx}", "fisher_indices.npy"),
            threshold       = args.fisher_threshold,
            sr              = args.sr,
            use_skfeature   = use_skfeature,
        )

    print(f"\n[Fisher] All {n_folds} fold(s) complete.")


if __name__ == '__main__':
    main()