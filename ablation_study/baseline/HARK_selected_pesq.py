"""
Measure PESQ for the HARK outputs selected by the existing WER evaluation.

The official HARK evaluators select one separated wav per target utterance and
write that choice in pipeline_whisper_*_official_hark_wer_details.csv. This
script reuses those selections, pairs them with the matching clean reference,
and writes PESQ details plus a JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
DEFAULT_RESULTS_DIR = BASELINE_ROOT / "results" / "HARK_n_sep+pf_IPD"
DEFAULT_DETAILS_CSV = DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_wer_details.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PESQ for selected HARK separated wavs from the WER details CSV."
    )
    parser.add_argument("--details_csv", type=Path, default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_selected_pesq_details.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_selected_pesq_summary.json",
    )
    parser.add_argument("--sample_rate", type=int, default=16000, choices=(8000, 16000))
    parser.add_argument(
        "--mode",
        choices=("wb", "nb"),
        default="wb",
        help="Use wb at 16000 Hz or nb at 8000 Hz. PESQ only supports these combinations.",
    )
    parser.add_argument(
        "--selected_column",
        type=str,
        default="selected_wav",
        help="Column containing the enhanced/selected wav path.",
    )
    parser.add_argument(
        "--mic_column",
        type=str,
        default="mic_file",
        help="Column containing the mixture wav path used for input PESQ. Leave present for PESQi.",
    )
    parser.add_argument(
        "--mic_channel",
        type=int,
        default=0,
        help="Zero-based mixture channel used for input PESQ and PESQi.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit rows for a quick check; 0 means all rows.")
    return parser.parse_args()


def path_from_csv(value: str) -> Path:
    raw = (value or "").strip()
    if not raw:
        return Path()

    path = Path(raw)
    if path.exists():
        return path

    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        return Path(f"{drive}:\\{rest}")

    return path


def parse_int_field(row: Dict[str, str], field: str) -> int:
    value = row.get(field, "")
    if value == "":
        raise ValueError(f"Missing required column value: {field}")
    return int(float(value))


def clean_path_for_row(row: Dict[str, str], clean_dir: Path) -> Path:
    fileid = parse_int_field(row, "fileid")
    speaker_id = parse_int_field(row, "speaker_id")
    gt_doa = parse_int_field(row, "gt_doa")
    return clean_dir / f"clean_fileid_{fileid}_doa{gt_doa}_spk{speaker_id}.wav"


def load_audio(path: Path, target_sr: int, channel: Optional[int] = None) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if channel is None:
        audio = wav[:, 0]
    else:
        if channel < 0 or channel >= wav.shape[1]:
            raise ValueError(f"Channel {channel} is out of range for {path} with {wav.shape[1]} channel(s).")
        audio = wav[:, channel]

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return audio, sr


def pesq_score(reference: np.ndarray, degraded: np.ndarray, sample_rate: int, mode: str) -> float:
    from pesq import pesq

    if mode == "wb" and sample_rate != 16000:
        raise ValueError("Wideband PESQ requires --sample_rate 16000.")
    if mode == "nb" and sample_rate != 8000:
        raise ValueError("Narrowband PESQ requires --sample_rate 8000.")

    min_len = min(len(reference), len(degraded))
    if min_len <= 0:
        raise ValueError("Cannot score empty audio.")
    return float(pesq(sample_rate, reference[:min_len], degraded[:min_len], mode))


def mean_optional(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def load_rows(path: Path, max_items: int) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_items > 0:
        return rows[:max_items]
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], input_fieldnames: Iterable[str]) -> None:
    extra_fields = [
        "clean_file",
        "pesq_mode",
        "sample_rate",
        "input_wb_pesq",
        "wb_pesq",
        "wb_pesq_i",
        "pesq_error",
    ]
    fieldnames = list(input_fieldnames)
    for field in extra_fields:
        if field not in fieldnames:
            fieldnames.append(field)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    scored = [row for row in rows if row.get("wb_pesq") not in (None, "")]
    errors = [row for row in rows if row.get("pesq_error")]
    return {
        "details_csv": str(args.details_csv),
        "clean_dir": str(args.clean_dir),
        "out_csv": str(args.out_csv),
        "sample_rate": args.sample_rate,
        "pesq_mode": args.mode,
        "total_rows": len(rows),
        "scored_rows": len(scored),
        "error_rows": len(errors),
        "mean_input_wb_pesq": mean_optional(rows, "input_wb_pesq"),
        "mean_wb_pesq": mean_optional(rows, "wb_pesq"),
        "mean_wb_pesqi": mean_optional(rows, "wb_pesq_i"),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "wb" and args.sample_rate != 16000:
        raise ValueError("--mode wb requires --sample_rate 16000.")
    if args.mode == "nb" and args.sample_rate != 8000:
        raise ValueError("--mode nb requires --sample_rate 8000.")
    if not args.details_csv.is_file():
        raise FileNotFoundError(f"Details CSV not found: {args.details_csv}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean directory not found: {args.clean_dir}")

    rows = load_rows(args.details_csv, args.max_items)
    if rows and args.selected_column not in rows[0]:
        raise KeyError(f"Selected wav column not found: {args.selected_column}")

    output_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows, start=1):
        out_row: Dict[str, object] = dict(row)
        clean_path = clean_path_for_row(row, args.clean_dir)
        selected_path = path_from_csv(row.get(args.selected_column, ""))
        mic_path = path_from_csv(row.get(args.mic_column, ""))
        out_row.update(
            {
                "clean_file": str(clean_path),
                "pesq_mode": args.mode,
                "sample_rate": args.sample_rate,
                "input_wb_pesq": None,
                "wb_pesq": None,
                "wb_pesq_i": None,
                "pesq_error": "",
            }
        )

        try:
            if not clean_path.is_file():
                raise FileNotFoundError(f"Clean wav not found: {clean_path}")
            if not selected_path.is_file():
                raise FileNotFoundError(f"Selected wav not found: {selected_path}")

            clean, _ = load_audio(clean_path, args.sample_rate)
            enhanced, _ = load_audio(selected_path, args.sample_rate)
            enhanced_pesq = pesq_score(clean, enhanced, args.sample_rate, args.mode)
            out_row["wb_pesq"] = enhanced_pesq

            input_pesq: Optional[float] = None
            if mic_path.is_file():
                noisy, _ = load_audio(mic_path, args.sample_rate, channel=args.mic_channel)
                input_pesq = pesq_score(clean, noisy, args.sample_rate, args.mode)
                out_row["input_wb_pesq"] = input_pesq

            if input_pesq is not None:
                out_row["wb_pesq_i"] = enhanced_pesq - input_pesq
        except Exception as exc:
            out_row["pesq_error"] = f"{type(exc).__name__}: {exc}"

        output_rows.append(out_row)
        if idx % 25 == 0 or idx == len(rows):
            print(f"Processed {idx}/{len(rows)} rows")

    input_fieldnames = rows[0].keys() if rows else []
    write_csv(args.out_csv, output_rows, input_fieldnames)

    summary = summarize(output_rows, args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== SELECTED HARK PESQ SUMMARY =====")
    print(f"Scored rows: {summary['scored_rows']} / {summary['total_rows']}")
    print(f"Mean input PESQ: {summary['mean_input_wb_pesq']}")
    print(f"Mean selected PESQ: {summary['mean_wb_pesq']}")
    print(f"Mean PESQi: {summary['mean_wb_pesqi']}")
    print(f"Saved details: {args.out_csv}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
