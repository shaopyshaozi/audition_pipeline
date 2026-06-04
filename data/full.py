from pathlib import Path
import numpy as np
import soundfile as sf

# =========================
# Config
# =========================
MIC_DIR = Path(
    "/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/"
    "data/dataset_4mic_3spk_4s_full/Eval/mic"
)

OUTPUT_DIR = MIC_DIR.parent / "mic_4s_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLIP_SECONDS = 4.0

# =========================
# Split each wav into 4s clips
# =========================
wav_files = sorted(MIC_DIR.glob("*.wav"))

print(f"Input folder: {MIC_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Total wav files found: {len(wav_files)}")

for wav_path in wav_files:
    audio, sr = sf.read(wav_path, always_2d=False)

    clip_len = int(CLIP_SECONDS * sr)
    total_len = len(audio)

    num_clips = int(np.ceil(total_len / clip_len))

    print(f"\nProcessing {wav_path.name}")
    print(f"  sr={sr}, duration={total_len / sr:.2f}s, clips={num_clips}")

    for i in range(num_clips):
        start = i * clip_len
        end = start + clip_len

        clip = audio[start:end]

        # Pad last clip to exactly 4 seconds
        if len(clip) < clip_len:
            pad_len = clip_len - len(clip)

            if audio.ndim == 1:
                pad = np.zeros(pad_len, dtype=audio.dtype)
            else:
                pad = np.zeros((pad_len, audio.shape[1]), dtype=audio.dtype)

            clip = np.concatenate([clip, pad], axis=0)

        output_name = f"{wav_path.stem}_{i + 1}{wav_path.suffix}"
        output_path = OUTPUT_DIR / output_name

        sf.write(output_path, clip, sr)

print("\nDone. All files have been split into 4-second clips.")