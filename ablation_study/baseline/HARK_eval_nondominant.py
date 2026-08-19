#!/usr/bin/env python3
"""
Evaluate non-dominant-speaker WER from already generated official-HARK outputs.

Default settings processed:
  results/HARK_n_sep_GT
  results/HARK_n_sep_IPD
  results/HARK_n_sep+pf_GT
  results/HARK_n_sep+pf_IPD

Selection rule:
  * GT-DOA HARK folders: choose the enhanced wav whose name contains the
    target speaker id, e.g. doa*_spk2.wav or doa*_spk3.wav.
  * IPDNet-DOA HARK folders: choose the enhanced wav whose predicted DOA in
    the filename is closest to the target speaker's GT DOA.

This script does not run HARK. It only evaluates separated wavs that already
exist under each setting's official_hark_outputs/fileid_* directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import resample_poly
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
OFFLINE_ROOT = PROJECT_ROOT / "ablation_study" / "baseline"
DEFAULT_SETTINGS = (
    "HARK_n_sep_GT",
    "HARK_n_sep_IPD",
    "HARK_n_sep+pf_GT",
    "HARK_n_sep+pf_IPD",
)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def edit_distance_words(ref_words: Sequence[str], hyp_words: Sequence[str]) -> int:
    n, m = len(ref_words), len(hyp_words)
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
    if not ref_words:
        return (0.0 if not hyp_words else 1.0, len(hyp_words), 0)
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


def parse_predicted_doa(path_or_name: Path | str) -> Optional[int]:
    match = re.search(r"preddoa(\d+)", Path(path_or_name).name)
    return int(match.group(1)) if match else None


def parse_source_index(path_or_name: Path | str) -> Optional[int]:
    match = re.search(r"src(\d+)", Path(path_or_name).name)
    return int(match.group(1)) - 1 if match else None


def circular_angle_error_deg(pred_deg: float, gt_deg: float) -> float:
    return float(abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0))


def load_mono_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return wav, sr


def load_first_channel(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return wav, sr


def si_sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    scale = np.dot(pred, target) / (np.dot(target, target) + eps)
    projection = scale * target
    noise = pred - projection
    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)))


def sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    error = target - pred
    return float(10.0 * np.log10((np.sum(target**2) + eps) / (np.sum(error**2) + eps)))


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: Path
    clean_path: Optional[Path]


@dataclass
class EvalRecord:
    setting: str
    fileid: int
    speaker_id: int
    duration_sec: float
    gt_doa: int
    selected_wav: str
    selection_strategy: str
    selected_source_index: Optional[int]
    predicted_doa: Optional[float]
    doa_error_deg: Optional[float]
    separated_wav_count: int
    gt_text_file: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    input_sdr: Optional[float]
    sdr: Optional[float]
    sdr_i: Optional[float]
    input_si_sdr: Optional[float]
    si_sdr: Optional[float]
    si_sdr_i: Optional[float]
    reference: str
    hypothesis: str
    whisper_sec: float
    whisper_rtf: float


def elapsed_seconds(device: str, fn):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def load_references(
    text_dir: Path,
    clean_dir: Path,
    target_speaker_ids: Sequence[int],
) -> Dict[int, List[TargetReference]]:
    clean_by_key: Dict[Tuple[int, int, int], Path] = {}
    for clean_path in sorted(clean_dir.glob("clean_fileid_*_doa*_spk*.wav")):
        fileid = parse_fileid(clean_path)
        speaker_id = parse_speaker_id(clean_path)
        doa = parse_doa(clean_path)
        clean_by_key[(fileid, speaker_id, doa)] = clean_path

    refs_by_fileid: Dict[int, List[TargetReference]] = {}
    target_set = set(target_speaker_ids)
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk*.txt")):
        speaker_id = parse_speaker_id(text_path)
        if speaker_id not in target_set:
            continue
        fileid = parse_fileid(text_path)
        doa = parse_doa(text_path)
        refs_by_fileid.setdefault(fileid, []).append(
            TargetReference(
                fileid=fileid,
                speaker_id=speaker_id,
                gt_doa=doa,
                text_path=text_path,
                clean_path=clean_by_key.get((fileid, speaker_id, doa)),
            )
        )

    for refs in refs_by_fileid.values():
        refs.sort(key=lambda ref: (ref.speaker_id, ref.gt_doa, ref.text_path.name))
    return refs_by_fileid


def find_mic_path(mic_dir: Path, fileid: int) -> Optional[Path]:
    matches = sorted(mic_dir.glob(f"mic_fileid_{fileid}_*.wav"))
    return matches[0] if matches else None


def setting_kind(setting: str) -> str:
    if setting.endswith("_GT"):
        return "gt"
    if setting.endswith("_IPD"):
        return "ipd"
    raise ValueError(f"Cannot infer setting kind from: {setting}")


def find_scene_dirs(setting_dir: Path, max_items: int) -> List[Path]:
    root = setting_dir / "official_hark_outputs"
    if not root.exists():
        return []
    scene_dirs = sorted(
        [path for path in root.glob("fileid_*") if path.is_dir()],
        key=lambda path: parse_fileid(path),
    )
    if max_items > 0:
        return scene_dirs[:max_items]
    return scene_dirs


def find_enhanced_wavs(scene_dir: Path) -> List[Path]:
    return sorted(path for path in scene_dir.glob("enhanced_fileid_*.wav") if path.is_file())


def select_gt_wav(scene_dir: Path, ref: TargetReference) -> Tuple[Path, Optional[int], Optional[int], Optional[float]]:
    expected = scene_dir / f"enhanced_fileid_{ref.fileid}_doa{ref.gt_doa}_spk{ref.speaker_id}.wav"
    if expected.exists():
        return expected, ref.speaker_id - 1, None, None

    matches = sorted(scene_dir.glob(f"enhanced_fileid_{ref.fileid}_doa*_spk{ref.speaker_id}.wav"))
    if matches:
        selected = matches[0]
        return selected, ref.speaker_id - 1, None, None
    raise FileNotFoundError(f"No GT-DOA HARK wav found for fileid={ref.fileid}, spk{ref.speaker_id}")


def select_ipd_wav(scene_dir: Path, ref: TargetReference) -> Tuple[Path, Optional[int], Optional[int], Optional[float]]:
    candidates: List[Tuple[float, Path, Optional[int], int]] = []
    for wav_path in find_enhanced_wavs(scene_dir):
        pred_doa = parse_predicted_doa(wav_path)
        if pred_doa is None:
            continue
        candidates.append(
            (
                circular_angle_error_deg(pred_doa, ref.gt_doa),
                wav_path,
                parse_source_index(wav_path),
                pred_doa,
            )
        )
    if not candidates:
        raise FileNotFoundError(f"No IPDNet-DOA HARK wav found for fileid={ref.fileid}")
    doa_error, selected_wav, source_index, pred_doa = sorted(candidates, key=lambda item: (item[0], item[1].name))[0]
    return selected_wav, source_index, pred_doa, doa_error


def compute_audio_quality_metrics(
    *,
    enhanced: np.ndarray,
    clean_path: Optional[Path],
    noisy_ref: Optional[np.ndarray],
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    empty = {
        "input_sdr": None,
        "sdr": None,
        "sdr_i": None,
        "input_si_sdr": None,
        "si_sdr": None,
        "si_sdr_i": None,
    }
    if clean_path is None:
        return empty

    clean, clean_sr = load_mono_audio(clean_path, target_sr=sample_rate)
    if clean_sr != sample_rate or len(clean) == 0 or len(enhanced) == 0:
        return empty

    min_len = min(len(enhanced), len(clean))
    enhanced = enhanced[:min_len]
    clean = clean[:min_len]

    output_sdr = sdr_db(enhanced, clean)
    output_si_sdr = si_sdr_db(enhanced, clean)
    metrics = {
        "input_sdr": None,
        "sdr": output_sdr,
        "sdr_i": None,
        "input_si_sdr": None,
        "si_sdr": output_si_sdr,
        "si_sdr_i": None,
    }

    if noisy_ref is not None and len(noisy_ref) > 0:
        noisy_min_len = min(len(noisy_ref), len(clean))
        noisy = noisy_ref[:noisy_min_len]
        clean_for_noisy = clean[:noisy_min_len]
        input_sdr = sdr_db(noisy, clean_for_noisy)
        input_si_sdr = si_sdr_db(noisy, clean_for_noisy)
        metrics.update(
            {
                "input_sdr": input_sdr,
                "sdr_i": output_sdr - input_sdr,
                "input_si_sdr": input_si_sdr,
                "si_sdr_i": output_si_sdr - input_si_sdr,
            }
        )
    return metrics


def parse_speaker_ids(value: str) -> List[int]:
    ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ids:
        raise argparse.ArgumentTypeError("At least one speaker id is required.")
    return ids


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def summarize_rows(rows: Sequence[EvalRecord]) -> Dict[str, object]:
    total_edits = sum(int(row.edit_distance or 0) for row in rows)
    total_ref_words = sum(int(row.ref_words or 0) for row in rows)
    wers = [float(row.wer) for row in rows if row.wer is not None]
    return {
        "evaluated_utterances": len(rows),
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words else None,
        "mean_sample_wer": float(np.mean(wers)) if wers else None,
        "total_edit_distance": total_edits,
        "total_ref_words": total_ref_words,
        "mean_sdr": safe_mean(row.sdr for row in rows),
        "mean_sdri": safe_mean(row.sdr_i for row in rows),
        "mean_si_sdr": safe_mean(row.si_sdr for row in rows),
        "mean_sisdri": safe_mean(row.si_sdr_i for row in rows),
        "mean_doa_error_deg": safe_mean(row.doa_error_deg for row in rows),
        "mean_whisper_sec": safe_mean(row.whisper_sec for row in rows),
        "mean_whisper_rtf": safe_mean(row.whisper_rtf for row in rows),
    }


def fmt_metric(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def evaluate_setting(
    *,
    args: argparse.Namespace,
    setting: str,
    whisper_model,
    refs_by_fileid: Dict[int, List[TargetReference]],
) -> Tuple[List[EvalRecord], Dict[str, int]]:
    setting_dir = args.results_root / setting
    kind = setting_kind(setting)
    scene_dirs = find_scene_dirs(setting_dir, args.max_items)
    records: List[EvalRecord] = []
    skipped = {
        "no_scene_dirs": int(not scene_dirs),
        "no_target_refs": 0,
        "no_selected_wav": 0,
        "asr_failed": 0,
    }

    desc = f"Eval {setting}"
    for scene_dir in tqdm(scene_dirs, desc=desc, unit="scene"):
        fileid = parse_fileid(scene_dir)
        target_refs = refs_by_fileid.get(fileid, [])
        if not target_refs:
            skipped["no_target_refs"] += 1
            continue

        mic_path = find_mic_path(args.mic_dir, fileid)
        noisy_ref = None
        if mic_path is not None and mic_path.exists():
            try:
                noisy_ref, _ = load_first_channel(mic_path, target_sr=args.sample_rate)
            except Exception as exc:
                print(f"fileid={fileid}: failed to load mic audio for metrics: {exc}")

        separated_wav_count = len(find_enhanced_wavs(scene_dir))
        for ref in target_refs:
            try:
                if kind == "gt":
                    selected_wav, source_index, pred_doa, doa_error = select_gt_wav(scene_dir, ref)
                    strategy = "gt_spk_filename"
                else:
                    selected_wav, source_index, pred_doa, doa_error = select_ipd_wav(scene_dir, ref)
                    strategy = f"nearest_predicted_doa_to_spk{ref.speaker_id}_gt_doa"
                enhanced, sr = load_mono_audio(selected_wav, target_sr=args.sample_rate)
            except Exception as exc:
                skipped["no_selected_wav"] += 1
                print(f"{setting} fileid={fileid}, spk{ref.speaker_id}: selection failed: {exc}")
                continue

            ref_text = ref.text_path.read_text(encoding="utf-8").strip()

            def run_asr():
                return whisper_model.transcribe(
                    enhanced,
                    language=args.language,
                    fp16=args.whisper_device.startswith("cuda"),
                )

            try:
                asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
            except Exception as exc:
                skipped["asr_failed"] += 1
                print(f"{setting} fileid={fileid}, spk{ref.speaker_id}: Whisper failed: {exc}")
                continue

            hyp_text = asr_out.get("text", "").strip()
            sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)
            duration_sec = len(enhanced) / float(sr)
            metrics = compute_audio_quality_metrics(
                enhanced=enhanced,
                clean_path=ref.clean_path,
                noisy_ref=noisy_ref,
                sample_rate=sr,
            )
            records.append(
                EvalRecord(
                    setting=setting,
                    fileid=fileid,
                    speaker_id=ref.speaker_id,
                    duration_sec=duration_sec,
                    gt_doa=ref.gt_doa,
                    selected_wav=str(selected_wav),
                    selection_strategy=strategy,
                    selected_source_index=source_index,
                    predicted_doa=pred_doa,
                    doa_error_deg=doa_error,
                    separated_wav_count=separated_wav_count,
                    gt_text_file=str(ref.text_path),
                    wer=sample_wer,
                    edit_distance=dist,
                    ref_words=ref_word_count,
                    input_sdr=metrics["input_sdr"],
                    sdr=metrics["sdr"],
                    sdr_i=metrics["sdr_i"],
                    input_si_sdr=metrics["input_si_sdr"],
                    si_sdr=metrics["si_sdr"],
                    si_sdr_i=metrics["si_sdr_i"],
                    reference=ref_text,
                    hypothesis=hyp_text,
                    whisper_sec=whisper_sec,
                    whisper_rtf=whisper_sec / duration_sec if duration_sec > 0 else 0.0,
                )
            )
    return records, skipped


def build_grouped_summary(rows: Sequence[EvalRecord]) -> Dict[str, object]:
    by_setting = {}
    by_setting_speaker = {}
    for setting in sorted({row.setting for row in rows}):
        setting_rows = [row for row in rows if row.setting == setting]
        by_setting[setting] = summarize_rows(setting_rows)
        for speaker_id in sorted({row.speaker_id for row in setting_rows}):
            key = f"{setting}:spk{speaker_id}"
            by_setting_speaker[key] = summarize_rows(
                [row for row in setting_rows if row.speaker_id == speaker_id]
            )
    return {
        "overall": summarize_rows(rows),
        "by_setting": by_setting,
        "by_setting_speaker": by_setting_speaker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate spk2/spk3 WER from existing official-HARK separated wavs."
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--results_root", type=Path, default=OFFLINE_ROOT / "results")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / "HARK_n_nondominant_wer")
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--target_speaker_ids", type=parse_speaker_ids, default=parse_speaker_ids("2,3"))
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_items", type=int, default=0, help="Limit fileid folders per setting; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(whisper, "load_model"):
        raise RuntimeError(
            "The imported 'whisper' package does not provide load_model. "
            "Install OpenAI Whisper in this environment with: pip install -U openai-whisper"
        )
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")

    refs_by_fileid = load_references(args.text_dir, args.clean_dir, args.target_speaker_ids)
    print(f"Loaded non-dominant references for {len(refs_by_fileid)} scenes.")
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    all_records: List[EvalRecord] = []
    skipped_by_setting = {}
    for setting in args.settings:
        setting_records, skipped = evaluate_setting(
            args=args,
            setting=setting,
            whisper_model=whisper_model,
            refs_by_fileid=refs_by_fileid,
        )
        all_records.extend(setting_records)
        skipped_by_setting[setting] = skipped

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_hark_nondominant_wer_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in all_records:
            writer.writerow(asdict(row))

    summary = build_grouped_summary(all_records)
    summary.update(
        {
            "settings": args.settings,
            "target_speaker_ids": args.target_speaker_ids,
            "mic_dir": str(args.mic_dir),
            "clean_dir": str(args.clean_dir),
            "text_dir": str(args.text_dir),
            "results_root": str(args.results_root),
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "max_items": args.max_items,
            "skipped_by_setting": skipped_by_setting,
        }
    )
    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_hark_nondominant_wer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== HARK NON-DOMINANT SPEAKER WER SUMMARY =====")
    print("\nBy setting:")
    for setting in args.settings:
        setting_summary = summary["by_setting"].get(setting)
        if setting_summary is None:
            print(f"{setting}: n=0, corpus WER=N/A, mean WER=N/A")
            continue
        print(
            f"{setting}: n={setting_summary['evaluated_utterances']}, "
            f"corpus WER={fmt_metric(setting_summary['corpus_wer'])}, "
            f"mean WER={fmt_metric(setting_summary['mean_sample_wer'])}"
        )

    print("\nBy setting and speaker:")
    for setting in args.settings:
        for speaker_id in args.target_speaker_ids:
            key = f"{setting}:spk{speaker_id}"
            speaker_summary = summary["by_setting_speaker"].get(key)
            if speaker_summary is None:
                print(f"{setting} spk{speaker_id}: n=0, corpus WER=N/A, mean WER=N/A")
                continue
            print(
                f"{setting} spk{speaker_id}: "
                f"n={speaker_summary['evaluated_utterances']}, "
                f"corpus WER={fmt_metric(speaker_summary['corpus_wer'])}, "
                f"mean WER={fmt_metric(speaker_summary['mean_sample_wer'])}"
            )
    print(f"Saved details: {details_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
