"""
Run Whisper ASR over enhanced audio files and report Word Error Rate (WER).

Expected inputs:
- Enhanced audio folder:
    enhanced_fileid_<sceneid>_doa<angle>_spk<k>.wav
- Ground-truth text folder:
    text_fileid_<sceneid>_doa<angle>_spk<k>.txt
"""

from __future__ import annotations

import argparse
import csv
import re
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample_poly
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """
    Lightweight normalization for fairer WER comparison.
    """
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def edit_distance_words(ref_words: List[str], hyp_words: List[str]) -> int:
    """
    Classic Levenshtein distance at word level.
    """
    n = len(ref_words)
    m = len(hyp_words)

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,         # deletion
                dp[i, j - 1] + 1,         # insertion
                dp[i - 1, j - 1] + cost,  # substitution
            )
    return int(dp[n, m])


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    """
    Returns (wer_value, edit_distance, ref_word_count).
    """
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    n_ref = len(ref_words)
    if n_ref == 0:
        return (0.0 if len(hyp_words) == 0 else 1.0, len(hyp_words), 0)

    dist = edit_distance_words(ref_words, hyp_words)
    return dist / n_ref, dist, n_ref


def enhanced_to_text_path(enhanced_path: Path, text_root: Path) -> Path:
    """
    Map:
        enhanced_fileid_<sceneid>_doa<angle>_spk<k>.wav
    to:
        text_fileid_<sceneid>_doa<angle>_spk<k>.txt
    """
    stem = enhanced_path.stem
    if not stem.startswith("enhanced_fileid_"):
        raise ValueError(f"Unexpected enhanced filename: {enhanced_path.name}")

    text_stem = stem.replace("enhanced_fileid_", "text_fileid_", 1)
    return text_root / f"{text_stem}.txt"


def load_audio_mono(audio_path: Path, channel: int = 0) -> np.ndarray:
    """
    Load audio and return mono waveform at 16 kHz.
    If multi-channel, select the requested channel.
    """
    wav, sr = sf.read(str(audio_path), always_2d=True)  # [T, C]

    if channel < 0 or channel >= wav.shape[1]:
        raise ValueError(
            f"Requested channel={channel}, but audio has {wav.shape[1]} channels: {audio_path}"
        )

    mono = wav[:, channel].astype(np.float32)

    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        mono = resample_poly(mono, up, down).astype(np.float32)

    return mono


@dataclass
class SampleResult:
    text_file: str
    audio_file: str
    reference: str
    hypothesis: str
    wer: float
    edit_distance: int
    ref_words: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER on enhanced audio files.")
    parser.add_argument(
        "--enhanced_dir",
        type=str,
        default=r"/mnt/d/邵鹏远/UCL/博1/code/DSENet/eval/enhanced_wer_4mics",
        help="Folder containing enhanced wav files",
    )
    parser.add_argument(
        "--text_dir",
        type=str,
        default=r"/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/data/dataset_4mic_3spk/Eval/text",
        help="Folder containing ground-truth text files",
    )
    parser.add_argument("--model", type=str, default="turbo", help="Whisper model name")
    parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")
    parser.add_argument("--language", type=str, default="en", help="Whisper language hint")
    parser.add_argument("--channel", type=int, default=0, help="Channel index if audio is multi-channel")
    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Optional cap for quick tests; 0 means use all files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/results",
        help="Directory to store CSV results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    enhanced_dir = Path(args.enhanced_dir)
    text_dir = Path(args.text_dir)

    if not enhanced_dir.is_dir():
        raise FileNotFoundError(f"Enhanced audio folder not found: {enhanced_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {text_dir}")

    enhanced_files = sorted(enhanced_dir.glob("*.wav"))
    if args.max_items and args.max_items > 0:
        enhanced_files = enhanced_files[:args.max_items]

    if len(enhanced_files) == 0:
        raise FileNotFoundError(f"No enhanced wav files found in: {enhanced_dir}")

    print(f"Loading Whisper model={args.model} on device={args.device}...")
    model = whisper.load_model(args.model, device=args.device)

    sample_results: List[SampleResult] = []
    total_edits = 0
    total_ref_words = 0
    missing_text = 0

    for idx, audio_path in enumerate(tqdm(enhanced_files, desc="Evaluating", unit="utt"), start=1):
        text_path = enhanced_to_text_path(audio_path, text_dir)

        if not text_path.is_file():
            missing_text += 1
            print(f"[{idx}/{len(enhanced_files)}] missing text: {text_path.name}")
            continue

        ref_text = text_path.read_text(encoding="utf-8").strip()
        audio = load_audio_mono(audio_path, args.channel)

        result = model.transcribe(audio, language=args.language, fp16=False)
        hyp_text = result.get("text", "").strip()

        sample_wer, dist, ref_words = wer(ref_text, hyp_text)
        total_edits += dist
        total_ref_words += ref_words

        sample_results.append(
            SampleResult(
                text_file=text_path.name,
                audio_file=audio_path.name,
                reference=ref_text,
                hypothesis=hyp_text,
                wer=sample_wer,
                edit_distance=dist,
                ref_words=ref_words,
            )
        )

    corpus_wer = (total_edits / total_ref_words) if total_ref_words > 0 else 0.0
    mean_sample_wer = float(np.mean([x.wer for x in sample_results])) if sample_results else 0.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"enhanced_whisper_{args.model}_wer_details.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text_file",
                "audio_file",
                "wer",
                "edit_distance",
                "ref_words",
                "reference",
                "hypothesis",
            ],
        )
        writer.writeheader()
        for row in sample_results:
            writer.writerow(asdict(row))

    print("\n===== WER SUMMARY =====")
    print(f"Whisper model: {args.model}, device: {args.device}")
    print(f"Enhanced files requested: {len(enhanced_files)}")
    print(f"Evaluated items: {len(sample_results)}")
    print(f"Missing text pairs: {missing_text}")
    print(f"Corpus WER: {corpus_wer:.4f}")
    print(f"Mean sample WER: {mean_sample_wer:.4f}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()