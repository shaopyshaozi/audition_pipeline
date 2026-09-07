"""
Compute PESQ for speaker 1 only.

PESQ needs a clean reference and a degraded/enhanced signal. This script uses
the spk1 clean files from data/dataset_4mic_3spk/Eval/clean and compares them
with the HARK selected wavs stored in the WER details CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from HARK_selected_pesq import (
    clean_path_for_row,
    load_audio,
    path_from_csv,
    pesq_score,
    write_csv,
)


BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
DEFAULT_RESULTS_DIR = BASELINE_ROOT / "results" / "HARK_n_loc+sep+pf"
DEFAULT_DETAILS_CSV = DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_wer_details.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute wideband PESQ for spk1 clean references against selected HARK wavs."
    )
    parser.add_argument("--details_csv", type=Path, default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_selected_pesq_spk1_details.csv",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "pipeline_whisper_small_official_hark_selected_pesq_spk1_summary.json",
    )
    parser.add_argument("--speaker_id", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=16000, choices=(8000, 16000))
    parser.add_argument("--mode", choices=("wb", "nb"), default="wb")
    parser.add_argument("--selected_column", type=str, default="selected_wav")
    parser.add_argument("--mic_column", type=str, default="mic_file")
    parser.add_argument("--mic_channel", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=0)
    return parser.parse_args()


def load_rows(path: Path, max_items: int) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if max_items > 0:
        return rows[:max_items]
    return rows


def speaker_matches(row: Dict[str, str], speaker_id: int) -> bool:
    try:
        return int(float(row.get("speaker_id", ""))) == speaker_id
    except ValueError:
        return False


def mean_optional(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def summarize(rows: Sequence[Dict[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    scored = [row for row in rows if row.get("wb_pesq") not in (None, "")]
    errors = [row for row in rows if row.get("pesq_error")]
    return {
        "details_csv": str(args.details_csv),
        "clean_dir": str(args.clean_dir),
        "speaker_id": args.speaker_id,
        "clean_glob": f"clean_fileid_*_spk{args.speaker_id}.wav",
        "out_csv": str(args.out_csv),
        "sample_rate": args.sample_rate,
        "pesq_mode": args.mode,
        "total_spk_rows": len(rows),
        "scored_rows": len(scored),
        "error_rows": len(errors),
        "mean_input_wb_pesq": mean_optional(rows, "input_wb_pesq"),
        "mean_wb_pesq": mean_optional(rows, "wb_pesq"),
        "mean_wb_pesqi": mean_optional(rows, "wb_pesq_i"),
    }


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


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

    all_rows = load_rows(args.details_csv, args.max_items)
    rows = [row for row in all_rows if speaker_matches(row, args.speaker_id)]
    if not rows:
        raise RuntimeError(f"No rows found for speaker_id={args.speaker_id} in {args.details_csv}")
    if args.selected_column not in rows[0]:
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
            if f"_spk{args.speaker_id}.wav" not in clean_path.name:
                raise ValueError(f"Clean file is not spk{args.speaker_id}: {clean_path}")
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
            print(f"Processed {idx}/{len(rows)} spk{args.speaker_id} rows")

    input_fieldnames: Iterable[str] = rows[0].keys()
    write_csv(args.out_csv, output_rows, input_fieldnames)

    summary = summarize(output_rows, args)
    write_summary(args.summary_json, summary)

    print(f"\n===== HARK SELECTED PESQ SPK{args.speaker_id} SUMMARY =====")
    print(f"Scored rows: {summary['scored_rows']} / {summary['total_spk_rows']}")
    print(f"Mean input PESQ: {summary['mean_input_wb_pesq']}")
    print(f"Mean selected PESQ: {summary['mean_wb_pesq']}")
    print(f"Mean PESQi: {summary['mean_wb_pesqi']}")
    print(f"Saved details: {args.out_csv}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
