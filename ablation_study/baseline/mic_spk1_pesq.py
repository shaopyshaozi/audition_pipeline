"""
Compute PESQ for raw Eval/mic files against spk1 clean references.

The Eval/mic folder has multiple wavs per fileid. By default this script picks
one mic wav per fileid by matching the spk1 clean DOA:

    clean_fileid_10_doa3_spk1.wav -> mic_fileid_10_doa3_3spk.wav

PESQ is then computed between the spk1 clean reference and one selected mic
channel, channel 0 by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
DEFAULT_OUT_DIR = BASELINE_ROOT / "results" / "mic_spk1_pesq"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PESQ for one raw mic file per fileid against spk1 clean.")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_DIR / "mic_spk1_pesq_details.csv")
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_OUT_DIR / "mic_spk1_pesq_summary.json")
    parser.add_argument("--speaker_id", type=int, default=1)
    parser.add_argument("--mic_channel", type=int, default=0)
    parser.add_argument("--sample_rate", type=int, default=16000, choices=(8000, 16000))
    parser.add_argument("--mode", choices=("wb", "nb"), default="wb")
    parser.add_argument(
        "--selection_strategy",
        choices=("match_clean_doa", "lowest_doa"),
        default="match_clean_doa",
        help="How to select one mic wav when there are multiple wavs for a fileid.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit clean spk rows for a quick test; 0 means all.")
    return parser.parse_args()


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


def collect_mics_by_fileid(mic_dir: Path) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in sorted(mic_dir.glob("mic_fileid_*_doa*_3spk.wav"), key=lambda p: (parse_fileid(p), parse_doa(p))):
        grouped.setdefault(parse_fileid(path), []).append(path)
    return grouped


def select_mic_path(
    *,
    mic_by_fileid: Dict[int, List[Path]],
    fileid: int,
    clean_doa: int,
    strategy: str,
) -> Path:
    candidates = mic_by_fileid.get(fileid, [])
    if not candidates:
        raise FileNotFoundError(f"No mic wavs found for fileid={fileid}")

    if strategy == "match_clean_doa":
        for path in candidates:
            if parse_doa(path) == clean_doa:
                return path
        raise FileNotFoundError(f"No mic wav found for fileid={fileid}, doa={clean_doa}")

    if strategy == "lowest_doa":
        return sorted(candidates, key=lambda path: (parse_doa(path), path.name))[0]

    raise ValueError(f"Unknown selection strategy: {strategy}")


def mean_optional(rows: Sequence[Dict[str, object]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "") and np.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else None


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "fileid",
        "speaker_id",
        "gt_doa",
        "clean_file",
        "selected_mic_file",
        "selection_strategy",
        "mic_channel",
        "sample_rate",
        "pesq_mode",
        "mic_pesq",
        "pesq_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "wb" and args.sample_rate != 16000:
        raise ValueError("--mode wb requires --sample_rate 16000.")
    if args.mode == "nb" and args.sample_rate != 8000:
        raise ValueError("--mode nb requires --sample_rate 8000.")
    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic directory not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean directory not found: {args.clean_dir}")

    clean_paths = sorted(
        args.clean_dir.glob(f"clean_fileid_*_doa*_spk{args.speaker_id}.wav"),
        key=lambda path: (parse_fileid(path), parse_doa(path), path.name),
    )
    if args.max_items > 0:
        clean_paths = clean_paths[: args.max_items]
    if not clean_paths:
        raise FileNotFoundError(f"No spk{args.speaker_id} clean wavs found in {args.clean_dir}")

    mic_by_fileid = collect_mics_by_fileid(args.mic_dir)
    rows: List[Dict[str, object]] = []
    for idx, clean_path in enumerate(clean_paths, start=1):
        fileid = parse_fileid(clean_path)
        speaker_id = parse_speaker_id(clean_path)
        clean_doa = parse_doa(clean_path)
        row: Dict[str, object] = {
            "fileid": fileid,
            "speaker_id": speaker_id,
            "gt_doa": clean_doa,
            "clean_file": str(clean_path),
            "selected_mic_file": "",
            "selection_strategy": args.selection_strategy,
            "mic_channel": args.mic_channel,
            "sample_rate": args.sample_rate,
            "pesq_mode": args.mode,
            "mic_pesq": None,
            "pesq_error": "",
        }

        try:
            mic_path = select_mic_path(
                mic_by_fileid=mic_by_fileid,
                fileid=fileid,
                clean_doa=clean_doa,
                strategy=args.selection_strategy,
            )
            row["selected_mic_file"] = str(mic_path)
            clean, _ = load_audio(clean_path, args.sample_rate)
            mic_audio, _ = load_audio(mic_path, args.sample_rate, channel=args.mic_channel)
            row["mic_pesq"] = pesq_score(clean, mic_audio, args.sample_rate, args.mode)
        except Exception as exc:
            row["pesq_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)
        if idx % 25 == 0 or idx == len(clean_paths):
            print(f"Processed {idx}/{len(clean_paths)} spk{args.speaker_id} clean rows")

    write_csv(args.out_csv, rows)
    error_rows = [row for row in rows if row.get("pesq_error")]
    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
        "speaker_id": args.speaker_id,
        "clean_glob": f"clean_fileid_*_doa*_spk{args.speaker_id}.wav",
        "selection_strategy": args.selection_strategy,
        "mic_channel": args.mic_channel,
        "sample_rate": args.sample_rate,
        "pesq_mode": args.mode,
        "total_rows": len(rows),
        "scored_rows": len(rows) - len(error_rows),
        "error_rows": len(error_rows),
        "mean_mic_pesq": mean_optional(rows, "mic_pesq"),
        "out_csv": str(args.out_csv),
    }
    write_summary(args.summary_json, summary)

    print(f"\n===== MIC PESQ SPK{args.speaker_id} SUMMARY =====")
    print(f"Scored rows: {summary['scored_rows']} / {summary['total_rows']}")
    print(f"Mean mic PESQ: {summary['mean_mic_pesq']}")
    print(f"Saved details: {args.out_csv}")
    print(f"Saved summary: {args.summary_json}")


if __name__ == "__main__":
    main()
