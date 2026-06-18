from pathlib import Path
import math
import re

import numpy as np
import soundfile as sf


# =========================
# Config
# =========================
DATA_ROOT = Path(__file__).resolve().parent
ROOT = DATA_ROOT / "dataset_4mic_3spk_4s_full" / "Eval"

CLEAN_DIR = ROOT / "clean"
OUTPUT_CLEAN_DIR = ROOT / "clean_4s"
OUTPUT_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

CLIP_SECONDS = 4.0


# =========================
# Helpers
# =========================
def get_file_id(path: Path):
    m = re.search(r"fileid[_-]?(\d+)", path.stem)
    if m:
        return int(m.group(1))

    m = re.search(r"\d+", path.stem)
    if m:
        return int(m.group(0))

    return None


def is_spk1(path: Path):
    name = path.stem.lower()
    return "spk1" in name or "speaker1" in name


def pad_clip(clip: np.ndarray, clip_len: int, dtype) -> np.ndarray:
    if len(clip) >= clip_len:
        return clip

    pad_len = clip_len - len(clip)
    if clip.ndim == 1:
        pad = np.zeros(pad_len, dtype=dtype)
    else:
        pad = np.zeros((pad_len, clip.shape[1]), dtype=dtype)

    return np.concatenate([clip, pad], axis=0)


# =========================
# Split spk1 clean files
# =========================
spk1_clean_paths = [
    clean_path
    for clean_path in sorted(CLEAN_DIR.glob("*.wav"))
    if is_spk1(clean_path)
]

print(f"Found spk1 clean files: {len(spk1_clean_paths)}")

total_clips = 0

for clean_path in spk1_clean_paths:
    file_id = get_file_id(clean_path)
    if file_id is None:
        print(f"Warning: cannot extract file_id from {clean_path.name}")

    audio, sr = sf.read(clean_path, always_2d=False)
    clip_len = int(CLIP_SECONDS * sr)
    num_clips = math.ceil(len(audio) / clip_len)

    print(
        f"\n{clean_path.name}: "
        f"length={len(audio) / sr:.2f}s, "
        f"clips={num_clips}"
    )

    for i in range(num_clips):
        start = i * clip_len
        end = start + clip_len

        clip = audio[start:end]
        clip = pad_clip(clip, clip_len, audio.dtype)

        output_name = f"{clean_path.stem}_{i + 1}{clean_path.suffix}"
        output_path = OUTPUT_CLEAN_DIR / output_name

        sf.write(output_path, clip, sr)
        total_clips += 1

print(f"\nDone. Saved {total_clips} clean spk1 clips to {OUTPUT_CLEAN_DIR}")
