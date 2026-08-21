#!/usr/bin/env python3
"""
Evaluate saved IPDNet->DSENet enhanced audio for the dominant speaker.

This script does not rerun IPDNet, DSENet, or Whisper. It joins the saved
enhanced wavs with the existing WER/DoA details CSV, then compares each spk1
enhanced file against:

    clean target audio      -> output quality
    noisy mixture channel   -> input quality and improvement

Default paths target:
    offline/IPDNET/results/pipeline_full/pipeline_enhanced
    offline/IPDNET/results/pipeline_full/pipeline_whisper_small_wer_details.csv
    data/dataset_4mic_3spk/Eval/{mic,clean}
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DSE_ROOT = PROJECT_ROOT / "Models" / "DSE"
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk" / "Eval"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "pipeline_full"

sys.path.insert(0, str(DSE_ROOT))
from models.utils.metrics import cal_metrics_functional  # noqa: E402


ENHANCED_RE = re.compile(
    r"^enhanced_fileid_(?P<fileid>\d+)_gt(?P<gt_doa>\d+)_pred(?P<pred_doa>\d+)_spk(?P<spk>\d+)\.wav$"
)


def parse_metric_list(value: str) -> List[str]:
    metrics = [item.strip() for item in value.split(",") if item.strip()]
    if not metrics:
        raise argparse.ArgumentTypeError("Expected a comma-separated metric list.")
    return metrics


def parse_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value)))
    except ValueError:
        return None


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        parsed = float(str(value))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def sum_numeric(values: Iterable[Optional[float]]) -> float:
    return float(sum(float(value) for value in values if value is not None and math.isfinite(float(value))))


def scalar_or_none(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_mono_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    mono = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return mono, sr


def load_mixture_channel(path: Path, target_sr: int, mic_channel: int) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    mic_idx = mic_channel - 1
    if mic_idx < 0 or mic_idx >= wav.shape[1]:
        raise ValueError(f"Invalid --mic_channel {mic_channel} for {wav.shape[1]}-channel file: {path}")
    channel = wav[:, mic_idx]
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        channel = resample_poly(channel, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return channel, sr


def build_enhanced_index(enhanced_dir: Path) -> Dict[Tuple[int, int, int], Path]:
    index: Dict[Tuple[int, int, int], Path] = {}
    for path in sorted(enhanced_dir.glob("enhanced_fileid_*_gt*_pred*_spk*.wav")):
        match = ENHANCED_RE.match(path.name)
        if not match:
            continue
        key = (
            int(match.group("fileid")),
            int(match.group("gt_doa")),
            int(match.group("spk")),
        )
        index[key] = path
    return index


def find_clean_path(clean_dir: Path, fileid: int, gt_doa: int, speaker_id: int) -> Optional[Path]:
    path = clean_dir / f"clean_fileid_{fileid}_doa{gt_doa}_spk{speaker_id}.wav"
    return path if path.is_file() else None


def find_mic_path(mic_dir: Path, row: Dict[str, str], fileid: int, gt_doa: int) -> Optional[Path]:
    mic_name = row.get("mic_file", "")
    if mic_name:
        direct = mic_dir / Path(mic_name).name
        if direct.is_file():
            return direct
    matches = sorted(mic_dir.glob(f"mic_fileid_{fileid}_doa{gt_doa}_*.wav"))
    return matches[0] if matches else None


def compute_audio_metrics(
    *,
    enhanced: np.ndarray,
    clean: np.ndarray,
    noisy_ref: np.ndarray,
    sample_rate: int,
    metric_list: Sequence[str],
) -> Dict[str, Optional[float]]:
    min_len = min(len(enhanced), len(clean), len(noisy_ref))
    if min_len <= 0:
        return {
            "input_sdr": None,
            "sdr": None,
            "sdr_i": None,
            "input_si_sdr": None,
            "si_sdr": None,
            "si_sdr_i": None,
            "input_wb_pesq": None,
            "wb_pesq": None,
            "wb_pesq_i": None,
        }

    pred_t = torch.from_numpy(enhanced[:min_len]).float().unsqueeze(0)
    clean_t = torch.from_numpy(clean[:min_len]).float().unsqueeze(0)
    noisy_t = torch.from_numpy(noisy_ref[:min_len]).float().unsqueeze(0)

    metrics, input_metrics, imp_metrics = cal_metrics_functional(
        list(metric_list),
        preds=pred_t,
        target=clean_t,
        original=noisy_t,
        fs=sample_rate,
        device_only=None,
    )

    return {
        "input_sdr": scalar_or_none(input_metrics.get("input_sdr")),
        "sdr": scalar_or_none(metrics.get("sdr")),
        "sdr_i": scalar_or_none(imp_metrics.get("sdr_i")),
        "input_si_sdr": scalar_or_none(input_metrics.get("input_si_sdr")),
        "si_sdr": scalar_or_none(metrics.get("si_sdr")),
        "si_sdr_i": scalar_or_none(imp_metrics.get("si_sdr_i")),
        "input_wb_pesq": scalar_or_none(input_metrics.get("input_wb_pesq")),
        "wb_pesq": scalar_or_none(metrics.get("wb_pesq")),
        "wb_pesq_i": scalar_or_none(imp_metrics.get("wb_pesq_i")),
    }


def read_target_rows(path: Path, target_speaker_id: int, max_items: int) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if parse_int(row.get("spk")) == target_speaker_id
        ]
    rows.sort(key=lambda row: (parse_int(row.get("fileid")) or -1, parse_int(row.get("gt_doa")) or -1))
    if max_items > 0:
        rows = rows[:max_items]
    return rows


def summarize(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    edit_distance = sum_numeric(parse_float(row.get("edit_distance")) for row in rows)
    ref_words = sum_numeric(parse_float(row.get("ref_words")) for row in rows)
    doa_errors = [parse_float(row.get("doa_error_deg")) for row in rows]
    doa_errors_finite = [value for value in doa_errors if value is not None]

    summary = {
        "wer_details_csv": str(args.wer_details),
        "enhanced_dir": str(args.enhanced_dir),
        "clean_dir": str(args.clean_dir),
        "mic_dir": str(args.mic_dir),
        "target_speaker_id": args.target_speaker_id,
        "sample_rate": args.sample_rate,
        "mic_channel": args.mic_channel,
        "metric_list": list(args.metric_list),
        "evaluated_items": len(rows),
        "missing_enhanced": int(sum(1 for row in rows if row.get("missing_reason") == "missing_enhanced")),
        "missing_clean": int(sum(1 for row in rows if row.get("missing_reason") == "missing_clean")),
        "missing_mic": int(sum(1 for row in rows if row.get("missing_reason") == "missing_mic")),
        "metric_failures": int(sum(1 for row in rows if row.get("metrics_ok") == 0)),
        "corpus_wer": edit_distance / ref_words if ref_words > 0 else None,
        "mean_sample_wer": mean(parse_float(row.get("wer")) for row in rows),
        "total_edit_distance": edit_distance,
        "total_ref_words": ref_words,
        "mean_doa_error_deg": mean(doa_errors),
        "median_doa_error_deg": float(median(doa_errors_finite)) if doa_errors_finite else None,
        "doa_within_10_deg_count": int(sum(1 for value in doa_errors_finite if value <= 10.0)),
        "doa_within_20_deg_count": int(sum(1 for value in doa_errors_finite if value <= 20.0)),
        "doa_within_30_deg_count": int(sum(1 for value in doa_errors_finite if value <= 30.0)),
        "mean_input_sdr": mean(parse_float(row.get("input_sdr")) for row in rows),
        "mean_sdr": mean(parse_float(row.get("sdr")) for row in rows),
        "mean_sdri": mean(parse_float(row.get("sdr_i")) for row in rows),
        "mean_input_si_sdr": mean(parse_float(row.get("input_si_sdr")) for row in rows),
        "mean_si_sdr": mean(parse_float(row.get("si_sdr")) for row in rows),
        "mean_sisdri": mean(parse_float(row.get("si_sdr_i")) for row in rows),
        "mean_input_wb_pesq": mean(parse_float(row.get("input_wb_pesq")) for row in rows),
        "mean_wb_pesq": mean(parse_float(row.get("wb_pesq")) for row in rows),
        "mean_wb_pesqi": mean(parse_float(row.get("wb_pesq_i")) for row in rows),
    }
    evaluated = len(doa_errors_finite)
    for threshold in (10, 20, 30):
        count = summary[f"doa_within_{threshold}_deg_count"]
        summary[f"doa_within_{threshold}_deg_percent"] = (100.0 * count / evaluated) if evaluated else None
    return summary


def write_summary_csv(path: Path, summary: Dict[str, object]) -> None:
    keys = [
        "evaluated_items",
        "corpus_wer",
        "mean_sample_wer",
        "mean_doa_error_deg",
        "median_doa_error_deg",
        "doa_within_10_deg_percent",
        "doa_within_20_deg_percent",
        "doa_within_30_deg_percent",
        "mean_input_sdr",
        "mean_sdr",
        "mean_sdri",
        "mean_input_si_sdr",
        "mean_si_sdr",
        "mean_sisdri",
        "mean_input_wb_pesq",
        "mean_wb_pesq",
        "mean_wb_pesqi",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerow({key: summary.get(key) for key in keys})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SDR/SI-SDR/WB-PESQ metrics for saved enhanced spk1 audio."
    )
    parser.add_argument("--wer_details", type=Path, default=DEFAULT_RESULTS_DIR / "pipeline_whisper_small_wer_details.csv")
    parser.add_argument("--enhanced_dir", type=Path, default=DEFAULT_RESULTS_DIR / "pipeline_enhanced")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "clean")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "mic")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output_prefix", type=str, default="pipeline_full_sisdr")
    parser.add_argument("--target_speaker_id", type=int, default=1)
    parser.add_argument("--mic_channel", type=int, default=1, help="1-based mixture channel used as noisy baseline.")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--metric_list", type=parse_metric_list, default=parse_metric_list("SDR,SI_SDR,WB_PESQ"))
    parser.add_argument("--max_items", type=int, default=0, help="Limit rows for quick testing; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wer_details.is_file():
        raise FileNotFoundError(f"WER/DoA details CSV not found: {args.wer_details}")
    if not args.enhanced_dir.is_dir():
        raise FileNotFoundError(f"Enhanced audio folder not found: {args.enhanced_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean audio folder not found: {args.clean_dir}")
    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mixture audio folder not found: {args.mic_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = args.out_dir / f"{args.output_prefix}_details.csv"
    summary_json = args.out_dir / f"{args.output_prefix}_summary.json"
    summary_csv = args.out_dir / f"{args.output_prefix}_summary.csv"

    enhanced_index = build_enhanced_index(args.enhanced_dir)
    target_rows = read_target_rows(args.wer_details, args.target_speaker_id, args.max_items)

    output_rows: List[Dict[str, object]] = []
    for row in tqdm(target_rows, desc=f"spk{args.target_speaker_id} saved enhanced metrics", unit="utt"):
        fileid = parse_int(row.get("fileid"))
        gt_doa = parse_int(row.get("gt_doa"))
        speaker_id = parse_int(row.get("spk"))
        if fileid is None or gt_doa is None or speaker_id is None:
            continue

        out_row: Dict[str, object] = dict(row)
        out_row.update(
            {
                "speaker_id": speaker_id,
                "enhanced_file": "",
                "clean_file": "",
                "mixture_file": "",
                "mic_channel": args.mic_channel,
                "metrics_ok": 0,
                "missing_reason": "",
                "duration_metric_sec": None,
            }
        )

        enhanced_path = enhanced_index.get((fileid, gt_doa, speaker_id))
        clean_path = find_clean_path(args.clean_dir, fileid, gt_doa, speaker_id)
        mic_path = find_mic_path(args.mic_dir, row, fileid, gt_doa)

        if enhanced_path is None:
            out_row["missing_reason"] = "missing_enhanced"
        elif clean_path is None:
            out_row["missing_reason"] = "missing_clean"
        elif mic_path is None:
            out_row["missing_reason"] = "missing_mic"
        else:
            enhanced, sr = load_mono_audio(enhanced_path, args.sample_rate)
            clean, _ = load_mono_audio(clean_path, args.sample_rate)
            noisy_ref, _ = load_mixture_channel(mic_path, args.sample_rate, args.mic_channel)
            metrics = compute_audio_metrics(
                enhanced=enhanced,
                clean=clean,
                noisy_ref=noisy_ref,
                sample_rate=sr,
                metric_list=args.metric_list,
            )
            out_row.update(metrics)
            out_row["enhanced_file"] = enhanced_path.name
            out_row["clean_file"] = clean_path.name
            out_row["mixture_file"] = mic_path.name
            out_row["duration_metric_sec"] = min(len(enhanced), len(clean), len(noisy_ref)) / float(sr)
            out_row["metrics_ok"] = int(any(value is not None for value in metrics.values()))

        output_rows.append(out_row)

    metric_columns = [
        "speaker_id",
        "enhanced_file",
        "clean_file",
        "mixture_file",
        "mic_channel",
        "duration_metric_sec",
        "input_sdr",
        "sdr",
        "sdr_i",
        "input_si_sdr",
        "si_sdr",
        "si_sdr_i",
        "input_wb_pesq",
        "wb_pesq",
        "wb_pesq_i",
        "metrics_ok",
        "missing_reason",
    ]
    fieldnames = list(target_rows[0].keys()) if target_rows else []
    for column in metric_columns:
        if column not in fieldnames:
            fieldnames.append(column)

    with detail_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = summarize(output_rows, args)
    summary["details_csv"] = str(detail_csv)
    summary["summary_csv"] = str(summary_csv)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    write_summary_csv(summary_csv, summary)

    print("\n===== SAVED ENHANCED SPK1 AUDIO METRICS =====")
    print(f"Items: {summary['evaluated_items']}")
    print(f"Corpus WER: {summary['corpus_wer']:.6f}")
    print(f"Mean sample WER: {summary['mean_sample_wer']:.6f}")
    print(f"Mean DoA error: {summary['mean_doa_error_deg']:.3f} deg")
    print(f"Mean SDRi: {summary['mean_sdri']:.6f}")
    print(f"Mean SI-SDRi: {summary['mean_sisdri']:.6f}")
    print(f"Mean WB-PESQ: {summary['mean_wb_pesq']:.6f}")
    print(f"Mean WB-PESQi: {summary['mean_wb_pesqi']:.6f}")
    print(f"Saved details: {detail_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
