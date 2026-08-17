"""
Offline no-enhancement Whisper WER baseline.

For each saved multichannel ReSpeaker scene, this script selects one fixed mic
channel from the noisy mixture and sends it directly to Whisper. It compares
the transcript with the dominant speaker reference text, spk1 by default.

There is intentionally no enhancement, separation, beamforming, clean-audio
matching, or audio-quality metric computation here.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
SCRIPT_STEM = Path(__file__).stem
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_seconds(device: str, fn):
    cuda_sync(device)
    start = time.perf_counter()
    value = fn()
    cuda_sync(device)
    return value, time.perf_counter() - start


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def edit_distance_words(ref_words: Sequence[str], hyp_words: Sequence[str]) -> int:
    n = len(ref_words)
    m = len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if len(ref_words) == 0:
        return (0.0 if len(hyp_words) == 0 else 1.0, len(hyp_words), 0)
    dist = edit_distance_words(ref_words, hyp_words)
    return dist / len(ref_words), dist, len(ref_words)


def parse_fileid(path_or_name: Path | str) -> int:
    match = re.search(r"fileid_(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse fileid from: {path_or_name}")
    return int(match.group(1))


def parse_doa(path_or_name: Path | str) -> int:
    match = re.search(r"doa(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse doa from: {path_or_name}")
    return int(match.group(1))


def parse_speaker_id(path_or_name: Path | str) -> int:
    match = re.search(r"spk(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse speaker id from: {path_or_name}")
    return int(match.group(1))


def unique_mic_files(mic_dir: Path, max_items: int) -> List[Path]:
    all_files = sorted(mic_dir.glob("*.wav"), key=lambda p: (parse_fileid(p), parse_doa(p), p.name))
    if max_items > 0:
        return all_files[:max_items]
    return all_files


def group_targets_by_fileid(mic_files: Iterable[Path]) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in mic_files:
        grouped.setdefault(parse_fileid(path), []).append(path)
    return grouped


def choose_representative_mic(target_paths: Sequence[Path]) -> Path:
    return sorted(target_paths, key=lambda p: (parse_doa(p), p.name))[0]


def load_noisy_channel(path: Path, mic_channel: int, sample_rate: int) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if sr != sample_rate:
        raise ValueError(
            f"No_enhanced.py does not resample audio. Expected {sample_rate} Hz, got {sr} Hz: {path}"
        )

    mic_idx = mic_channel - 1
    if mic_idx < 0 or mic_idx >= wav.shape[1]:
        raise ValueError(f"--mic_channel {mic_channel} is invalid for {wav.shape[1]}-channel file: {path}")
    return wav[:, mic_idx], sr


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: Path


def load_target_references(text_dir: Path, target_speaker_id: int) -> Dict[int, TargetReference]:
    references: Dict[int, TargetReference] = {}
    for text_path in sorted(text_dir.glob(f"text_fileid_*_doa*_spk{target_speaker_id}.txt")):
        fileid = parse_fileid(text_path)
        ref = TargetReference(
            fileid=fileid,
            speaker_id=parse_speaker_id(text_path),
            gt_doa=parse_doa(text_path),
            text_path=text_path,
        )
        if fileid not in references:
            references[fileid] = ref
    return references


@dataclass
class AsrTiming:
    fileid: int
    mic_file: str
    method: str
    speaker_id: int
    duration_sec: float
    gt_doa: int
    mic_channel: int
    gt_text_file: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    reference: str
    hypothesis: str
    whisper_sec: float
    total_sec: float
    whisper_rtf: float
    total_rtf: float
    under_realtime: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noisy-mixture channel -> Whisper WER baseline.")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--mic_channel",
        type=int,
        default=1,
        help="1-based ReSpeaker channel to transcribe directly. Default selects channel 1.",
    )
    parser.add_argument(
        "--target_speaker_id",
        type=int,
        default=1,
        help="Speaker text reference to compare against. Default is dominant spk1.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    return parser.parse_args()


def append_result_for_scene(
    *,
    args: argparse.Namespace,
    whisper_model,
    fileid: int,
    mic_path: Path,
    ref: TargetReference,
) -> AsrTiming:
    noisy_channel, sr = load_noisy_channel(mic_path, args.mic_channel, args.sample_rate)
    duration_sec = len(noisy_channel) / float(sr)

    def run_asr():
        return whisper_model.transcribe(
            noisy_channel,
            language=args.language,
            fp16=args.whisper_device.startswith("cuda"),
        )

    asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
    hyp_text = asr_out.get("text", "").strip()
    ref_text = ref.text_path.read_text(encoding="utf-8").strip()
    sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)

    return AsrTiming(
        fileid=fileid,
        mic_file=mic_path.name,
        method="NoEnhancement-NoisyMixture",
        speaker_id=ref.speaker_id,
        duration_sec=duration_sec,
        gt_doa=ref.gt_doa,
        mic_channel=args.mic_channel,
        gt_text_file=ref.text_path.name,
        wer=sample_wer,
        edit_distance=dist,
        ref_words=ref_word_count,
        reference=ref_text,
        hypothesis=hyp_text,
        whisper_sec=whisper_sec,
        total_sec=whisper_sec,
        whisper_rtf=whisper_sec / duration_sec,
        total_rtf=whisper_sec / duration_sec,
        under_realtime=int(whisper_sec < duration_sec),
    )


def summarize(rows: Sequence[AsrTiming]) -> Dict[str, float | int]:
    evaluated = [row for row in rows if row.wer is not None]
    total_edits = sum(int(row.edit_distance or 0) for row in evaluated)
    total_ref_words = sum(int(row.ref_words or 0) for row in evaluated)
    wers = [float(row.wer) for row in evaluated]
    return {
        "evaluated_utterances": len(evaluated),
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
        "mean_sample_wer": float(np.mean(wers)) if wers else 0.0,
        "mean_whisper_sec": float(np.mean([row.whisper_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_sec": float(np.mean([row.total_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_rtf": float(np.mean([row.total_rtf for row in evaluated])) if evaluated else 0.0,
        "under_realtime_count": int(sum(row.under_realtime for row in evaluated)),
        "under_realtime_rate": float(np.mean([row.under_realtime for row in evaluated])) if evaluated else 0.0,
    }


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if args.target_speaker_id < 1:
        raise ValueError("--target_speaker_id must be a positive speaker id for this baseline.")
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Whisper device: {args.whisper_device}")
    print(f"Whisper model: {args.whisper_model}")
    print("Enhancement: none")
    print(f"Mic channel: {args.mic_channel} (1-based)")
    print(f"Target speaker: spk{args.target_speaker_id}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Target text folder: {args.text_dir}")

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    references_by_fileid = load_target_references(args.text_dir, args.target_speaker_id)

    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print(f"Target text references: {len(references_by_fileid)}")
    if not references_by_fileid:
        raise FileNotFoundError(
            f"No text_fileid_*_doa*_spk{args.target_speaker_id}.txt references found in {args.text_dir}"
        )

    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    rows: List[AsrTiming] = []
    skipped_no_target_text = 0
    for fileid, target_paths in tqdm(grouped.items(), desc="NoEnhancement-ASR", unit="scene"):
        ref = references_by_fileid.get(fileid)
        if ref is None:
            skipped_no_target_text += 1
            print(f"fileid={fileid}: no spk{args.target_speaker_id} text reference, skipped.")
            continue

        mic_path = choose_representative_mic(target_paths)
        rows.append(
            append_result_for_scene(
                args=args,
                whisper_model=whisper_model,
                fileid=fileid,
                mic_path=mic_path,
                ref=ref,
            )
        )

    target_label = f"spk{args.target_speaker_id}"
    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_no_enhanced_wer_details_{target_label}.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(AsrTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    overall = summarize(rows)
    summary = {
        "mic_dir": str(args.mic_dir),
        "text_dir": str(args.text_dir),
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "enhancement": "none_noisy_mixture_channel",
        "sample_rate": args.sample_rate,
        "mic_channel": args.mic_channel,
        "target_speaker_id": args.target_speaker_id,
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "target_text_references": len(references_by_fileid),
        "evaluated_utterances": len(rows),
        "skipped_no_target_text": skipped_no_target_text,
        "overall": overall,
    }

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_no_enhanced_wer_summary_{target_label}.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== NO-ENHANCEMENT ASR SUMMARY =====")
    print(f"Target speaker: spk{args.target_speaker_id}")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Skipped scenes without target text refs: {summary['skipped_no_target_text']}")
    print(
        f"NoEnhancement-NoisyMixture: corpus WER={overall['corpus_wer']:.4f}, "
        f"mean sample WER={overall['mean_sample_wer']:.4f}, "
        f"mean total RTF={overall['mean_total_rtf']:.3f}"
    )
    print(f"Saved details: {details_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
