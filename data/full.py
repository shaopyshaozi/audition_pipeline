from pathlib import Path
import re
import math
import numpy as np
import soundfile as sf

# =========================
# Config
# =========================
ROOT = Path(
    "/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/"
    "data/dataset_4mic_3spk_4s_full/Eval"
)

MIC_DIR = ROOT / "mic"
CLEAN_DIR = ROOT / "clean"

OUTPUT_MIC_DIR = ROOT / "mic_4s_split_by_spk1"
OUTPUT_MIC_DIR.mkdir(parents=True, exist_ok=True)

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


# =========================
# Build spk1 clean length map
# =========================
spk1_length_map = {}

for clean_path in sorted(CLEAN_DIR.glob("*.wav")):
    if not is_spk1(clean_path):
        continue

    file_id = get_file_id(clean_path)
    if file_id is None:
        print(f"Warning: cannot extract file_id from {clean_path.name}")
        continue

    info = sf.info(clean_path)
    spk1_length_map[file_id] = {
        "frames": info.frames,
        "sr": info.samplerate,
        "path": clean_path,
    }

print(f"Found spk1 clean files: {len(spk1_length_map)}")

# =========================
# Group mic files by file_id
# =========================
mic_groups = {}

for mic_path in sorted(MIC_DIR.glob("*.wav")):
    file_id = get_file_id(mic_path)
    if file_id is None:
        print(f"Warning: cannot extract file_id from {mic_path.name}")
        continue

    mic_groups.setdefault(file_id, []).append(mic_path)

print(f"Found mic file_ids: {len(mic_groups)}")

# =========================
# Split mic files based on spk1 length
# =========================
for file_id, mic_paths in sorted(mic_groups.items()):
    if file_id not in spk1_length_map:
        print(f"Skip file_id {file_id}: no matching spk1 clean file found")
        continue

    spk1_frames = spk1_length_map[file_id]["frames"]
    spk1_sr = spk1_length_map[file_id]["sr"]

    clip_len = int(CLIP_SECONDS * spk1_sr)
    num_clips = math.ceil(spk1_frames / clip_len)

    print(
        f"\nfile_id {file_id}: "
        f"spk1 length={spk1_frames / spk1_sr:.2f}s, "
        f"clips={num_clips}"
    )

    for mic_path in mic_paths:
        audio, mic_sr = sf.read(mic_path, always_2d=False)

        if mic_sr != spk1_sr:
            raise ValueError(
                f"Sample rate mismatch for file_id {file_id}: "
                f"mic sr={mic_sr}, spk1 sr={spk1_sr}"
            )

        # Only keep mixture up to speaker 1 length
        audio = audio[:spk1_frames]

        for i in range(num_clips):
            start = i * clip_len
            end = start + clip_len

            clip = audio[start:end]

            # Pad remaining part with silence
            if len(clip) < clip_len:
                pad_len = clip_len - len(clip)

                if audio.ndim == 1:
                    pad = np.zeros(pad_len, dtype=audio.dtype)
                else:
                    pad = np.zeros((pad_len, audio.shape[1]), dtype=audio.dtype)

                clip = np.concatenate([clip, pad], axis=0)

            output_name = f"{mic_path.stem}_{i + 1}{mic_path.suffix}"
            output_path = OUTPUT_MIC_DIR / output_name

            sf.write(output_path, clip, mic_sr)

print("\nDone.")