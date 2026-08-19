#!/usr/bin/env python3
"""
Summarize IPDNet DOA accuracy from official-HARK separated wav filenames.

The script reads predicted DOAs from filenames like:
  enhanced_fileid_0_preddoa182_src2.wav

It compares those predictions with each speaker's ground-truth DOA from:
  data/dataset_4mic_3spk/Eval/text/text_fileid_*_doa*_spk*.txt

For every scene and speaker, the selected prediction is the predicted DOA
closest to that speaker's GT DOA. The output reports the percentage of scenes
whose selected prediction is within 10, 20, and 30 degrees for spk1/spk2/spk3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
DEFAULT_HARK_IPD_DIR = PROJECT_ROOT / "ablation_study" / "baseline" / "results" / "HARK_n_sep_IPD"


@dataclass(frozen=True)
class SceneSpeakerRef:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: str


@dataclass
class DoaDetail:
    fileid: int
    speaker_id: int
    gt_doa: int
    selected_predicted_doa: Optional[int]
    selected_source_index: Optional[int]
    doa_error_deg: Optional[float]
    candidate_predicted_doas: str
    candidate_wavs: str
    selected_wav: str


def parse_fileid(path_or_name: Path | str) -> int:
    match = re.search(r"fileid_(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse fileid from: {path_or_name}")
    return int(match.group(1))


def parse_gt_doa(path_or_name: Path | str) -> int:
    match = re.search(r"doa(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse GT DOA from: {path_or_name}")
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
    return int(match.group(1)) if match else None


def circular_angle_error_deg(pred_deg: float, gt_deg: float) -> float:
    return float(abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0))


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return items


def load_gt_refs(text_dir: Path, speaker_ids: Sequence[int]) -> Dict[int, List[SceneSpeakerRef]]:
    wanted = set(speaker_ids)
    refs_by_fileid: Dict[int, List[SceneSpeakerRef]] = {}
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk*.txt")):
        speaker_id = parse_speaker_id(text_path)
        if speaker_id not in wanted:
            continue
        fileid = parse_fileid(text_path)
        gt_doa = parse_gt_doa(text_path)
        refs_by_fileid.setdefault(fileid, []).append(
            SceneSpeakerRef(
                fileid=fileid,
                speaker_id=speaker_id,
                gt_doa=gt_doa,
                text_path=str(text_path),
            )
        )

    for refs in refs_by_fileid.values():
        refs.sort(key=lambda ref: (ref.speaker_id, ref.gt_doa))
    return refs_by_fileid


def find_scene_dirs(outputs_root: Path, max_items: int) -> List[Path]:
    scene_dirs = sorted(
        [path for path in outputs_root.glob("fileid_*") if path.is_dir()],
        key=lambda path: parse_fileid(path),
    )
    if max_items > 0:
        return scene_dirs[:max_items]
    return scene_dirs


def find_predicted_candidates(scene_dir: Path) -> List[Tuple[int, Optional[int], Path]]:
    candidates: List[Tuple[int, Optional[int], Path]] = []
    for wav_path in sorted(scene_dir.glob("enhanced_fileid_*_preddoa*_src*.wav")):
        pred_doa = parse_predicted_doa(wav_path)
        if pred_doa is None:
            continue
        candidates.append((pred_doa, parse_source_index(wav_path), wav_path))
    return candidates


def select_nearest_candidate(
    candidates: Sequence[Tuple[int, Optional[int], Path]],
    gt_doa: int,
) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[Path]]:
    if not candidates:
        return None, None, None, None
    ranked = sorted(
        (
            (circular_angle_error_deg(pred_doa, gt_doa), pred_doa, source_index, wav_path)
            for pred_doa, source_index, wav_path in candidates
        ),
        key=lambda item: (item[0], item[1], item[3].name),
    )
    error, pred_doa, source_index, wav_path = ranked[0]
    return pred_doa, source_index, error, wav_path


def build_details(
    refs_by_fileid: Dict[int, List[SceneSpeakerRef]],
    outputs_root: Path,
    max_items: int,
) -> Tuple[List[DoaDetail], Dict[str, int]]:
    records: List[DoaDetail] = []
    skipped = {
        "no_gt_refs": 0,
        "no_predicted_wavs": 0,
    }
    for scene_dir in find_scene_dirs(outputs_root, max_items):
        fileid = parse_fileid(scene_dir)
        refs = refs_by_fileid.get(fileid, [])
        if not refs:
            skipped["no_gt_refs"] += 1
            continue

        candidates = find_predicted_candidates(scene_dir)
        if not candidates:
            skipped["no_predicted_wavs"] += 1

        candidate_doas = ",".join(str(pred_doa) for pred_doa, _, _ in candidates)
        candidate_wavs = ";".join(str(path) for _, _, path in candidates)
        for ref in refs:
            pred_doa, source_index, error, selected_wav = select_nearest_candidate(candidates, ref.gt_doa)
            records.append(
                DoaDetail(
                    fileid=fileid,
                    speaker_id=ref.speaker_id,
                    gt_doa=ref.gt_doa,
                    selected_predicted_doa=pred_doa,
                    selected_source_index=source_index,
                    doa_error_deg=error,
                    candidate_predicted_doas=candidate_doas,
                    candidate_wavs=candidate_wavs,
                    selected_wav=str(selected_wav) if selected_wav is not None else "",
                )
            )
    return records, skipped


def safe_mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def safe_median(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.median(vals)) if vals else None


def summarize_by_speaker(records: Sequence[DoaDetail], thresholds: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    speaker_ids = sorted({row.speaker_id for row in records})
    for speaker_id in speaker_ids:
        speaker_rows = [row for row in records if row.speaker_id == speaker_id and row.doa_error_deg is not None]
        errors = [float(row.doa_error_deg) for row in speaker_rows]
        total = len(errors)
        summary: Dict[str, object] = {
            "speaker_id": speaker_id,
            "num_scenes": total,
            "mean_error_deg": safe_mean(errors),
            "median_error_deg": safe_median(errors),
        }
        for threshold in thresholds:
            within = sum(1 for err in errors if err <= threshold)
            summary[f"within_{threshold}_deg_count"] = within
            summary[f"within_{threshold}_deg_percent"] = (100.0 * within / total) if total else None
        rows.append(summary)
    return rows


def summarize_overall(records: Sequence[DoaDetail], thresholds: Sequence[int]) -> Dict[str, object]:
    valid = [row for row in records if row.doa_error_deg is not None]
    errors = [float(row.doa_error_deg) for row in valid]
    total = len(errors)
    summary: Dict[str, object] = {
        "num_speaker_examples": total,
        "mean_error_deg": safe_mean(errors),
        "median_error_deg": safe_median(errors),
    }
    for threshold in thresholds:
        within = sum(1 for err in errors if err <= threshold)
        summary[f"within_{threshold}_deg_count"] = within
        summary[f"within_{threshold}_deg_percent"] = (100.0 * within / total) if total else None
    return summary


def metric_text(value: object, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.2f}{suffix}"


def write_details_csv(path: Path, records: Sequence[DoaDetail]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DoaDetail.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def try_write_png_chart(path: Path, summary_rows: Sequence[Dict[str, object]], thresholds: Sequence[int]) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable, skipping PNG chart: {exc}")
        return False

    speakers = [f"spk{row['speaker_id']}" for row in summary_rows]
    x = np.arange(len(speakers))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    offsets = np.linspace(-width, width, num=len(thresholds))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for idx, threshold in enumerate(thresholds):
        values = [
            float(row.get(f"within_{threshold}_deg_percent") or 0.0)
            for row in summary_rows
        ]
        bars = ax.bar(
            x + offsets[idx],
            values,
            width,
            label=f"<= {threshold} deg",
            color=colors[idx % len(colors)],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=8)

    ax.set_title("IPDNet DOA Accuracy from HARK Enhanced Filenames")
    ax.set_ylabel("Scenes within threshold (%)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x)
    ax.set_xticklabels(speakers)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(thresholds), frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def write_svg_chart(path: Path, summary_rows: Sequence[Dict[str, object]], thresholds: Sequence[int]) -> None:
    width, height = 900, 520
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 70, 105
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    group_w = plot_w / max(1, len(summary_rows))
    bar_w = min(48, group_w / (len(thresholds) + 1.2))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    def y_for(value: float) -> float:
        return margin_top + plot_h - (value / 100.0) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">IPDNet DOA Accuracy from HARK Enhanced Filenames</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>',
    ]

    for tick in range(0, 101, 20):
        y = y_for(tick)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick}</text>')

    for group_idx, row in enumerate(summary_rows):
        speaker_label = f"spk{row['speaker_id']}"
        group_x = margin_left + group_idx * group_w + group_w / 2
        start_x = group_x - (len(thresholds) * bar_w) / 2
        parts.append(f'<text x="{group_x:.1f}" y="{height - 58}" text-anchor="middle" font-family="Arial" font-size="15">{speaker_label}</text>')
        for idx, threshold in enumerate(thresholds):
            value = float(row.get(f"within_{threshold}_deg_percent") or 0.0)
            x = start_x + idx * bar_w
            y = y_for(value)
            bar_h = margin_top + plot_h - y
            color = colors[idx % len(colors)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w * 0.41:.1f}" y="{max(y - 6, 54):.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.1f}%</text>')

    legend_x = margin_left + 180
    legend_y = height - 28
    for idx, threshold in enumerate(thresholds):
        x = legend_x + idx * 165
        color = colors[idx % len(colors)]
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="18" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 26}" y="{legend_y - 2}" font-family="Arial" font-size="13">&lt;= {threshold} deg</text>')
    parts.append('<text x="28" y="270" transform="rotate(-90 28 270)" text-anchor="middle" font-family="Arial" font-size="14">Scenes within threshold (%)</text>')
    parts.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IPD predicted-DOA threshold accuracy from HARK enhanced filenames."
    )
    parser.add_argument("--hark_ipd_dir", type=Path, default=DEFAULT_HARK_IPD_DIR)
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_HARK_IPD_DIR / "doa_error_summary")
    parser.add_argument("--speaker_ids", type=parse_int_list, default=parse_int_list("1,2,3"))
    parser.add_argument("--thresholds", type=parse_int_list, default=parse_int_list("10,20,30"))
    parser.add_argument("--max_items", type=int, default=0, help="Limit fileid folders for a quick test; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_root = args.hark_ipd_dir / "official_hark_outputs"
    if not outputs_root.exists():
        raise FileNotFoundError(f"Official HARK output folder not found: {outputs_root}")
    if not args.text_dir.exists():
        raise FileNotFoundError(f"GT text folder not found: {args.text_dir}")

    refs_by_fileid = load_gt_refs(args.text_dir, args.speaker_ids)
    details, skipped = build_details(refs_by_fileid, outputs_root, args.max_items)
    summary_rows = summarize_by_speaker(details, args.thresholds)
    overall = summarize_overall(details, args.thresholds)

    details_csv = args.out_dir / "ipd_doa_error_details.csv"
    summary_csv = args.out_dir / "ipd_doa_error_summary_by_speaker.csv"
    summary_json = args.out_dir / "ipd_doa_error_summary.json"
    png_chart = args.out_dir / "ipd_doa_error_thresholds.png"
    svg_chart = args.out_dir / "ipd_doa_error_thresholds.svg"

    write_details_csv(details_csv, details)
    write_summary_csv(summary_csv, summary_rows)
    summary_json.write_text(
        json.dumps(
            {
                "hark_ipd_dir": str(args.hark_ipd_dir),
                "text_dir": str(args.text_dir),
                "speaker_ids": args.speaker_ids,
                "thresholds": args.thresholds,
                "max_items": args.max_items,
                "overall": overall,
                "by_speaker": summary_rows,
                "skipped": skipped,
                "selection_rule": "For each speaker, use the predicted DOA from enhanced filename closest to that speaker's GT DOA.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    png_written = try_write_png_chart(png_chart, summary_rows, args.thresholds)
    write_svg_chart(svg_chart, summary_rows, args.thresholds)

    print("\n===== IPD DOA ERROR THRESHOLD SUMMARY =====")
    print(f"Evaluated speaker examples: {overall['num_speaker_examples']}")
    print(
        "Overall: "
        + ", ".join(
            f"<= {threshold} deg {metric_text(overall.get(f'within_{threshold}_deg_percent'), '%')}"
            for threshold in args.thresholds
        )
        + f", mean error {metric_text(overall.get('mean_error_deg'), ' deg')}"
    )
    for row in summary_rows:
        speaker = f"spk{row['speaker_id']}"
        parts = [
            f"<= {threshold} deg {metric_text(row.get(f'within_{threshold}_deg_percent'), '%')}"
            for threshold in args.thresholds
        ]
        parts.append(f"mean error {metric_text(row.get('mean_error_deg'), ' deg')}")
        parts.append(f"median error {metric_text(row.get('median_error_deg'), ' deg')}")
        parts.append(f"n={row['num_scenes']}")
        print(f"{speaker}: " + ", ".join(parts))

    print(f"\nSaved details: {details_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    if png_written:
        print(f"Saved PNG chart: {png_chart}")
    print(f"Saved SVG chart: {svg_chart}")


if __name__ == "__main__":
    main()
