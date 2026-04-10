# -*- coding: utf-8 -*-
"""
create_annotations.py
---------------------
Generates per-fold annotation files for the RAVDESS dataset using a
Leave-One-Actor-Out (LOAO) cross-validation protocol.

Expected directory layout:
    ROOT/
        ACTOR{01..24}/
            *_facecroppad.npy   (extracted + padded video frames)
            *_croppad.wav       (audio waveforms)

Split strategy — Leave-One-Actor-Out (LOAO)
-------------------------------------------
Actors are shuffled once with a fixed seed.  For each of the N folds:
    - test  : 1 actor  (the i-th actor in the shuffled order)
    - val   : 1 actor  (the next actor, wrapping cyclically)
    - train : all remaining actors

With 12 female actors this yields 12 folds of 10 / 1 / 1 (train / val / test),
giving the most stable per-actor evaluation possible for this dataset size.

Output format (semicolon-delimited, one sample per line):
    <npy_path>;<wav_path>;<emotion_label>;<split>
"""

import os
import random
import numpy as np


# ---------------------------------------------------------------------------
# CONFIGURATION  ← edit these
# ---------------------------------------------------------------------------

ROOT       = '/home/user/RAVDESS_BASELINE'   # root directory containing ACTOR* folders
SEED       = 42                     # controls actor shuffling — change to get different assignments

OUTPUT_DIR = '/home/user/MultimodalEmotionRecognition/ravdess_preprocessing'

# Actor sets — 1-based to match RAVDESS folder names (ACTOR01 → 1, ACTOR24 → 24)
# In RAVDESS: even actor numbers are female, odd actor numbers are male
ALL_ACTOR_IDS    = list(range(1, 25))                             # all 24 actors
FEMALE_ACTOR_IDS = [i for i in range(1, 25) if i % 2 == 0]       # ACTOR02,04,...,24 → female (12 actors)
MALE_ACTOR_IDS   = [i for i in range(1, 25) if i % 2 == 1]       # ACTOR01,03,...,23 → male   (12 actors)

ACTOR_IDS = MALE_ACTOR_IDS   # ← swap to MALE_ACTOR_IDS or ALL_ACTOR_IDS if needed

# LOAO: number of folds = number of actors (each actor is test exactly once)
N_FOLDS = len(ACTOR_IDS)


# ---------------------------------------------------------------------------
# RAVDESS FILENAME HELPERS
# ---------------------------------------------------------------------------

def parse_emotion_label(filepath: str) -> str | None:
    """
    Return the emotion code (3rd field) from a RAVDESS filename.

    RAVDESS format:
        [Modality]-[Channel]-[Emotion]-[Intensity]-[Statement]-[Repetition]-[Actor]

    Example:  01-01-03-01-02-01-05_facecroppad.npy  →  '03'
    """
    parts = os.path.basename(filepath).split('-')
    return parts[2] if len(parts) >= 3 else None


def actor_folder_to_index(folder_name: str) -> int | None:
    """
    Convert an ACTOR folder name to a 1-based actor index matching RAVDESS numbering.

    'ACTOR01' → 1,  'ACTOR24' → 24
    """
    try:
        return int(folder_name.split('ACTOR')[-1])
    except (ValueError, IndexError):
        return None


def npy_to_wav_path(npy_path: str) -> str:
    """
    Derive the corresponding .wav path from a _facecroppad.npy path.

    The audio file always starts with '03' and the rest of the identifier
    (excluding the first two characters and the '_face' suffix) is shared.

    Example:
        01-01-03-01-02-01-05_facecroppad.npy
        →  03-01-03-01-02-01-05_croppad.wav
    """
    base      = os.path.basename(npy_path)              # e.g. 01-01-..._facecroppad.npy
    stem      = base.split('_face')[0]                  # e.g. 01-01-...
    wav_name  = '03' + stem[2:] + '_croppad.wav'        # replace modality prefix with 03
    return os.path.join(os.path.dirname(npy_path), wav_name)


# ---------------------------------------------------------------------------
# FILE DISCOVERY
# ---------------------------------------------------------------------------

def collect_npy_files(root: str) -> list[str]:
    """
    Walk *root* and return a sorted list of all *_facecroppad.npy paths
    belonging to recognised ACTOR folders.
    """
    npy_files = []
    for folder in sorted(os.listdir(root)):
        actor_dir = os.path.join(root, folder)
        if not os.path.isdir(actor_dir):
            continue
        for fname in os.listdir(actor_dir):
            if os.path.basename(fname)[0:2] == '02':
                continue
            if fname.endswith('croppad.npy'):
                npy_files.append(os.path.join(actor_dir, fname))
    return sorted(npy_files)


# ---------------------------------------------------------------------------
# FOLD CREATION  (Leave-One-Actor-Out)
# ---------------------------------------------------------------------------

def create_folds(
    actor_ids: list[int],
    seed:      int,
) -> list[tuple[list, list, list]]:
    """
    Leave-One-Actor-Out (LOAO) cross-validation.

    Actors are shuffled once with *seed*, then for each fold i:
        test  = [ids[i]]               (1 actor)
        val   = [ids[(i+1) % n]]       (1 actor, next in shuffled order)
        train = all remaining actors   (n-2 actors)

    This guarantees:
        - Every actor appears as test in exactly one fold.
        - Every actor appears as val in exactly one fold.
        - Test and val actors never overlap.
        - No data leakage (actor-independent splits).

    Returns a list of (test_ids, val_ids, train_ids) tuples, one per fold.
    """
    ids = list(actor_ids)
    random.Random(seed).shuffle(ids)
    n   = len(ids)

    folds = []
    for i in range(n):
        test_ids  = [ids[i]]
        val_ids   = [ids[(i + 1) % n], ids[(i + 2) % n]]
        test_set  = set(test_ids)
        val_set   = set(val_ids)
        train_ids = [a for a in ids if a not in test_set and a not in val_set]
        folds.append((test_ids, val_ids, train_ids))

    return folds


# ---------------------------------------------------------------------------
# ANNOTATION WRITING
# ---------------------------------------------------------------------------

def write_fold_annotation(
    fold_idx:  int,
    test_ids:  list[int],
    val_ids:   list[int],
    train_ids: list[int],
    npy_files: list[str],
    actor_ids: set[int],
    output_dir: str,
) -> None:
    """Write a single fold's annotation file."""
    test_set  = set(test_ids)
    val_set   = set(val_ids)
    train_set = set(train_ids)

    out_path = os.path.join(output_dir, f'annotations_croppad_fold{fold_idx + 1}.txt')

    # Remove stale file from a previous run
    if os.path.exists(out_path):
        os.remove(out_path)

    counts = {'training': 0, 'validation': 0, 'testing': 0}

    with open(out_path, 'w') as f:
        for npy_path in npy_files:
            actor_folder = os.path.basename(os.path.dirname(npy_path))
            actor_idx    = actor_folder_to_index(actor_folder)

            # Skip actors not in the active set (e.g. male actors when using female only)
            if actor_idx is None or actor_idx not in actor_ids:
                continue

            label = parse_emotion_label(npy_path)
            if label is None:
                continue

            wav_path = npy_to_wav_path(npy_path)

            if actor_idx in train_set:
                split = 'training'
            elif actor_idx in val_set:
                split = 'validation'
            elif actor_idx in test_set:
                split = 'testing'
            else:
                continue

            f.write(f"{npy_path};{wav_path};{label};{split}\n")
            counts[split] += 1

    total = sum(counts.values())
    pct   = lambda k: f"{counts[k] / total:.0%}" if total else "0%"
    print(
        f"  Fold {fold_idx + 1} → {out_path}\n"
        f"    train={counts['training']} ({pct('training')})  "
        f"val={counts['validation']} ({pct('validation')})  "
        f"test={counts['testing']} ({pct('testing')})  "
        f"total={total}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    actor_set = set(ACTOR_IDS)
    n         = len(ACTOR_IDS)
    print(f"Dataset  : RAVDESS")
    print(f"Protocol : Leave-One-Actor-Out (LOAO)")
    print(f"Actors   : {n}  (indices {min(ACTOR_IDS)}–{max(ACTOR_IDS)})")
    print(f"Folds    : {N_FOLDS}  (1 test actor / 1 val actor / {n - 2} train actors per fold)")
    print(f"Root     : {ROOT}\n")

    # Discover all .npy files
    npy_files = collect_npy_files(ROOT)
    print(f"Found {len(npy_files)} .npy files\n")
    if not npy_files:
        print("ERROR: no .npy files found — check ROOT path.")
        return

    # Build LOAO folds
    folds = create_folds(ACTOR_IDS, SEED)

    print("--- Fold Actor Assignments ---")
    for i, (test, val, train) in enumerate(folds):
        print(
            f"  Fold {i + 1:>2}:  "
            f"test={sorted(test)}  "
            f"val={sorted(val)}  "
            f"train({len(train)})={sorted(train)}"
        )
    print()

    # Write annotation files
    print("--- Writing Annotation Files ---")
    for fold_idx, (test_ids, val_ids, train_ids) in enumerate(folds):
        write_fold_annotation(
            fold_idx, test_ids, val_ids, train_ids,
            npy_files, actor_set, OUTPUT_DIR,
        )


if __name__ == '__main__':
    main()