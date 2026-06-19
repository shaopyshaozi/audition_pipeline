from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_ROOT / "results" / "pipeline_streaming_4schunks_full" /"pipeline_streaming_small_scene_wer_1asr_4s_full.csv"
DOA_THRESHOLD_DEG = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise WER below a selected DOA-error threshold.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--doa_threshold_deg", type=float, default=DOA_THRESHOLD_DEG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    df["mean_selected_doa_error_deg"] = pd.to_numeric(
        df["mean_selected_doa_error_deg"], errors="coerce"
    )
    df["wer"] = pd.to_numeric(df["wer"], errors="coerce")

    valid_df = df.dropna(subset=["mean_selected_doa_error_deg", "wer"])
    subset = valid_df[valid_df["mean_selected_doa_error_deg"] <= args.doa_threshold_deg]

    total_samples = len(valid_df)
    selected_samples = len(subset)
    percentage = selected_samples / total_samples * 100 if total_samples > 0 else 0
    mean_wer = subset["wer"].mean()

    weighted_wer = None
    if {"edit_distance", "ref_words"}.issubset(valid_df.columns):
        total_edits = subset["edit_distance"].sum()
        total_words = subset["ref_words"].sum()
        if total_words > 0:
            weighted_wer = total_edits / total_words

    print(f"CSV path: {args.csv_path}")
    print(f"DOA threshold: {args.doa_threshold_deg:.1f} deg")
    print(f"Total samples: {total_samples}")
    print(f"Samples within threshold: {selected_samples}")
    print(f"Percentage within threshold: {percentage:.2f}%")
    print()
    print(f"Mean sample WER: {mean_wer:.4f}")

    if weighted_wer is not None:
        print(f"Corpus WER: {weighted_wer:.4f}")


if __name__ == "__main__":
    main()
