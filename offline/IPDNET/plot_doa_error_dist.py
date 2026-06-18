from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = SCRIPT_ROOT / "results" / "pipeline_realtime_1asr" / "pipeline_realtime_small_details_1asr.csv"
DEFAULT_OUTPUT_PATH = SCRIPT_ROOT / "selected_doa_error_hist_kde_0_180.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the selected DOA-error distribution.")
    parser.add_argument("--csv_path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_path)

    errors = pd.to_numeric(
        df["selected_doa_error_deg"],
        errors="coerce",
    ).dropna()
    errors = errors[(errors >= 0) & (errors <= 180)]

    if errors.empty:
        raise ValueError(f"No valid selected_doa_error_deg values found in {args.csv_path}")

    plt.figure(figsize=(8, 5))
    plt.hist(
        errors,
        bins=np.arange(0, 181, 5),
        density=True,
        alpha=0.4,
        edgecolor="black",
        label="Histogram",
    )

    if len(errors) > 1:
        x = np.linspace(0, 180, 1000)
        kde = gaussian_kde(errors)
        plt.plot(x, kde(x), linewidth=2, label="KDE")

    plt.xlim(0, 180)
    plt.xlabel("Selected DOA Error (degrees)")
    plt.ylabel("Density")
    plt.title("Selected DOA Error Distribution (0-180 deg)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_path, dpi=300)
    print(f"Saved plot: {args.output_path}")
    plt.show()


if __name__ == "__main__":
    main()
