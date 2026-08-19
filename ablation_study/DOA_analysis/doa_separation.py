#!/usr/bin/env python3
"""
Evaluate IPDNet -> DSENet -> Whisper on the controlled DoA-separation dataset.

The dataset is expected to use names like:
    mic_fileid_0_doa202_sep10_3spk.wav
    clean_fileid_0_doa202_sep10_spk1.wav
    text_fileid_0_doa202_sep10_spk1.txt

For each physical scene, IPDNet is run once. For each target speaker in the
scene, this script selects the predicted DoA nearest to that speaker's GT DoA,
runs DSENet with that predicted DoA, computes SDRi/SI-SDRi/PESQ, runs Whisper,
and summarizes all results by separation value.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import sys
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


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
OFFLINE_IPDNET_ROOT = PROJECT_ROOT / "offline" / "IPDNET"
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL" / "IPDNET"
DSE_ROOT = MODELS_ROOT / "DSE"
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_doa_sep_3spk" / "Eval"

sys.path.insert(0, str(OFFLINE_IPDNET_ROOT))
from pipeline_full import (  # noqa: E402
    circular_angle_diff,
    elapsed_seconds,
    enhance_doa_batch,
    load_dsenet,
    load_ipdnet,
    load_multichannel_audio,
    nearest_predicted_doa,
    postprocess_doa_from_tensors,
)

sys.path.insert(0, str(DSE_ROOT))
from models.utils.metrics import cal_metrics_functional  # noqa: E402


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
        raise ValueError(f"Could not parse DoA from: {path_or_name}")
    return int(match.group(1))


def parse_sep(path_or_name: Path | str) -> int:
    match = re.search(r"_sep(\d+)(?:_|\.)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse separation from: {path_or_name}")
    return int(match.group(1))


def parse_speaker_id(path_or_name: Path | str) -> int:
    match = re.search(r"spk(\d+)", Path(path_or_name).stem)
    if not match:
        raise ValueError(f"Could not parse speaker id from: {path_or_name}")
    return int(match.group(1))


def parse_metric_list(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated metric list.")
    return items


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated integer list.")
    return items


def load_mono_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = resample_poly(wav, up, down).astype(np.float32)
        sr = target_sr
    return wav, sr


def scalar_or_none(value) -> Optional[float]:
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
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


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
            "input_pesq": None,
            "pesq": None,
            "pesq_i": None,
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
        "input_pesq": scalar_or_none(input_metrics.get("input_wb_pesq")),
        "pesq": scalar_or_none(metrics.get("wb_pesq")),
        "pesq_i": scalar_or_none(imp_metrics.get("wb_pesq_i")),
    }


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    sep: int
    speaker_id: int
    gt_doa: int
    mic_path: Path
    clean_path: Path
    text_path: Path


@dataclass
class SeparationResult:
    fileid: int
    sep: int
    speaker_id: int
    gt_doa: int
    pred_doa: Optional[int]
    doa_error_deg: Optional[float]
    candidate_pred_doas: str
    mic_file: str
    clean_file: str
    text_file: str
    width: int
    duration_sec: float
    ref_channel: int
    input_sdr: Optional[float]
    sdr: Optional[float]
    sdr_i: Optional[float]
    input_si_sdr: Optional[float]
    si_sdr: Optional[float]
    si_sdr_i: Optional[float]
    input_pesq: Optional[float]
    pesq: Optional[float]
    pesq_i: Optional[float]
    wer: float
    edit_distance: int
    ref_words: int
    ipdnet_sec: float
    dsenet_sec: float
    whisper_sec: float
    total_sec: float
    enhanced_wav: str
    reference: str
    hypothesis: str


def find_single(candidates: Iterable[Path]) -> Optional[Path]:
    paths = sorted(candidates, key=lambda path: path.name)
    return paths[0] if paths else None


def load_target_references(
    *,
    mic_dir: Path,
    clean_dir: Path,
    text_dir: Path,
    max_scenes_per_sep: int,
    max_items: int,
) -> Tuple[List[TargetReference], Dict[str, int]]:
    refs: List[TargetReference] = []
    skipped = {
        "missing_mic": 0,
        "missing_clean": 0,
        "outside_max_scenes_per_sep": 0,
    }
    scenes_seen_by_sep: Dict[int, set[int]] = {}

    text_paths = sorted(
        text_dir.glob("text_fileid_*_doa*_sep*_spk*.txt"),
        key=lambda p: (parse_sep(p), parse_fileid(p), parse_speaker_id(p), parse_doa(p), p.name),
    )

    for text_path in text_paths:
        fileid = parse_fileid(text_path)
        sep = parse_sep(text_path)
        speaker_id = parse_speaker_id(text_path)
        gt_doa = parse_doa(text_path)

        sep_scenes = scenes_seen_by_sep.setdefault(sep, set())
        if fileid not in sep_scenes:
            if max_scenes_per_sep > 0 and len(sep_scenes) >= max_scenes_per_sep:
                skipped["outside_max_scenes_per_sep"] += 1
                continue
            sep_scenes.add(fileid)

        mic_path = find_single(mic_dir.glob(f"mic_fileid_{fileid}_doa{gt_doa}_sep{sep}_3spk.wav"))
        if mic_path is None:
            skipped["missing_mic"] += 1
            continue

        clean_path = find_single(clean_dir.glob(f"clean_fileid_{fileid}_doa{gt_doa}_sep{sep}_spk{speaker_id}.wav"))
        if clean_path is None:
            skipped["missing_clean"] += 1
            continue

        refs.append(
            TargetReference(
                fileid=fileid,
                sep=sep,
                speaker_id=speaker_id,
                gt_doa=gt_doa,
                mic_path=mic_path,
                clean_path=clean_path,
                text_path=text_path,
            )
        )
        if max_items > 0 and len(refs) >= max_items:
            break

    return refs, skipped


def group_refs_by_scene(refs: Sequence[TargetReference]) -> Dict[Tuple[int, int], List[TargetReference]]:
    groups: Dict[Tuple[int, int], List[TargetReference]] = {}
    for ref in refs:
        groups.setdefault((ref.sep, ref.fileid), []).append(ref)
    return dict(sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])))


def mean_optional(rows: Sequence[SeparationResult], attr: str) -> Optional[float]:
    values = [
        float(value)
        for row in rows
        for value in [getattr(row, attr)]
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: Sequence[SeparationResult]) -> List[Dict[str, Optional[float] | int | str]]:
    summary_rows: List[Dict[str, Optional[float] | int | str]] = []
    for sep in sorted({row.sep for row in rows}):
        sep_rows = [row for row in rows if row.sep == sep]
        total_edits = sum(row.edit_distance for row in sep_rows)
        total_ref_words = sum(row.ref_words for row in sep_rows)
        doa_errors = [
            float(row.doa_error_deg)
            for row in sep_rows
            if row.doa_error_deg is not None and np.isfinite(float(row.doa_error_deg))
        ]
        summary_rows.append(
            {
                "sep": sep,
                "evaluated_items": len(sep_rows),
                "unique_scenes": len({row.fileid for row in sep_rows}),
                "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
                "mean_sample_wer": float(np.mean([row.wer for row in sep_rows])) if sep_rows else 0.0,
                "mean_doa_error_deg": float(np.mean(doa_errors)) if doa_errors else None,
                "median_doa_error_deg": float(np.median(doa_errors)) if doa_errors else None,
                "doa_within_10_deg_percent": (
                    100.0 * sum(1 for err in doa_errors if err <= 10.0) / len(doa_errors)
                    if doa_errors
                    else None
                ),
                "doa_within_20_deg_percent": (
                    100.0 * sum(1 for err in doa_errors if err <= 20.0) / len(doa_errors)
                    if doa_errors
                    else None
                ),
                "doa_within_30_deg_percent": (
                    100.0 * sum(1 for err in doa_errors if err <= 30.0) / len(doa_errors)
                    if doa_errors
                    else None
                ),
                "mean_input_sdr": mean_optional(sep_rows, "input_sdr"),
                "mean_sdr": mean_optional(sep_rows, "sdr"),
                "mean_sdri": mean_optional(sep_rows, "sdr_i"),
                "mean_input_si_sdr": mean_optional(sep_rows, "input_si_sdr"),
                "mean_si_sdr": mean_optional(sep_rows, "si_sdr"),
                "mean_sisdri": mean_optional(sep_rows, "si_sdr_i"),
                "mean_input_pesq": mean_optional(sep_rows, "input_pesq"),
                "mean_pesq": mean_optional(sep_rows, "pesq"),
                "mean_pesqi": mean_optional(sep_rows, "pesq_i"),
                "mean_ipdnet_sec": mean_optional(sep_rows, "ipdnet_sec"),
                "mean_dsenet_sec": mean_optional(sep_rows, "dsenet_sec"),
                "mean_whisper_sec": mean_optional(sep_rows, "whisper_sec"),
                "mean_total_sec": mean_optional(sep_rows, "total_sec"),
            }
        )
    return summary_rows


def summarize_rows_by_sep_speaker(rows: Sequence[SeparationResult]) -> List[Dict[str, Optional[float] | int]]:
    summary_rows: List[Dict[str, Optional[float] | int]] = []
    for sep in sorted({row.sep for row in rows}):
        for speaker_id in sorted({row.speaker_id for row in rows if row.sep == sep}):
            subset = [row for row in rows if row.sep == sep and row.speaker_id == speaker_id]
            total_edits = sum(row.edit_distance for row in subset)
            total_ref_words = sum(row.ref_words for row in subset)
            summary_rows.append(
                {
                    "sep": sep,
                    "speaker_id": speaker_id,
                    "evaluated_items": len(subset),
                    "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
                    "mean_sample_wer": float(np.mean([row.wer for row in subset])) if subset else 0.0,
                    "mean_doa_error_deg": mean_optional(subset, "doa_error_deg"),
                    "mean_sdri": mean_optional(subset, "sdr_i"),
                    "mean_sisdri": mean_optional(subset, "si_sdr_i"),
                    "mean_pesq": mean_optional(subset, "pesq"),
                    "mean_pesqi": mean_optional(subset, "pesq_i"),
                }
            )
    return summary_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def try_write_plots(out_dir: Path, summary_rows: Sequence[Dict[str, object]]) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable, skipping plots: {exc}")
        return []

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    seps = [int(row["sep"]) for row in summary_rows]
    written: List[str] = []

    def values(name: str) -> List[float]:
        return [float(row[name]) if row.get(name) is not None else np.nan for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(seps, values("mean_doa_error_deg"), marker="o", label="Mean DOA error")
    ax.plot(seps, values("median_doa_error_deg"), marker="s", label="Median DOA error")
    ax.set_xlabel("Target-nearest-interferer separation (deg)")
    ax.set_ylabel("DOA error (deg)")
    ax.set_title("IPDNet DOA Error vs Source Separation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    doa_plot = plot_dir / "doa_error_vs_sep.png"
    fig.savefig(doa_plot)
    plt.close(fig)
    written.append(str(doa_plot))

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(seps, values("corpus_wer"), marker="o", label="Corpus WER")
    ax.plot(seps, values("mean_sample_wer"), marker="s", label="Mean sample WER")
    ax.set_xlabel("Target-nearest-interferer separation (deg)")
    ax.set_ylabel("WER")
    ax.set_title("Whisper WER vs Source Separation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    wer_plot = plot_dir / "wer_vs_sep.png"
    fig.savefig(wer_plot)
    plt.close(fig)
    written.append(str(wer_plot))

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax1.plot(seps, values("mean_sdri"), marker="o", label="SDRi")
    ax1.plot(seps, values("mean_sisdri"), marker="s", label="SI-SDRi")
    ax1.set_xlabel("Target-nearest-interferer separation (deg)")
    ax1.set_ylabel("Improvement (dB)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(seps, values("mean_pesq"), marker="^", color="#54a24b", label="PESQ")
    ax2.set_ylabel("PESQ")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False)
    ax1.set_title("Enhancement Metrics vs Source Separation")
    fig.tight_layout()
    metric_plot = plot_dir / "enhancement_metrics_vs_sep.png"
    fig.savefig(metric_plot)
    plt.close(fig)
    written.append(str(metric_plot))

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DOA prediction, enhancement metrics, and WER by source separation."
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "text")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "last-v1.ckpt")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_v13_99.ckpt")
    parser.add_argument("--out_dir", type=Path, default=EXPERIMENT_ROOT / "results" / "doa_separation")
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--vad_th", type=float, default=0.7)
    parser.add_argument("--min_points_per_source", type=int, default=3)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument(
        "--metrics",
        type=parse_metric_list,
        default=parse_metric_list("SDR,SI_SDR,WB_PESQ"),
        help="Comma-separated metrics passed to cal_metrics_functional.",
    )
    parser.add_argument(
        "--dse_batch_size",
        type=int,
        default=0,
        help="How many target DOAs to enhance per DSENet pass. 0 means all targets in a scene.",
    )
    parser.add_argument(
        "--max_scenes_per_sep",
        type=int,
        default=0,
        help="Limit physical scenes per sep for a quick test; 0 means all.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on target references; 0 means no cap.")
    parser.add_argument(
        "--summary_speakers",
        type=parse_int_list,
        default=parse_int_list("1,2"),
        help="Speaker IDs used for the main by-separation summary. Default: 1,2.",
    )
    parser.add_argument("--save_enhanced", action="store_true", help="Save enhanced wavs under out_dir/enhanced.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if not args.ipd_ckpt.is_file():
        raise FileNotFoundError(f"IPDNet checkpoint not found: {args.ipd_ckpt}")
    if not args.dse_ckpt.is_file():
        raise FileNotFoundError(f"DSENet checkpoint not found: {args.dse_ckpt}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for IPDNet/DSENet, but torch.cuda.is_available() is False.")
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA for Whisper, but torch.cuda.is_available() is False.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_root = args.out_dir / "enhanced"
    if args.save_enhanced:
        enhanced_root.mkdir(parents=True, exist_ok=True)

    refs, skipped = load_target_references(
        mic_dir=args.mic_dir,
        clean_dir=args.clean_dir,
        text_dir=args.text_dir,
        max_scenes_per_sep=args.max_scenes_per_sep,
        max_items=args.max_items,
    )
    if not refs:
        raise RuntimeError("No target references were found. Check dataset path and filename format.")

    groups = group_refs_by_scene(refs)

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"Mic folder: {args.mic_dir}")
    print(f"Clean folder: {args.clean_dir}")
    print(f"Text folder: {args.text_dir}")
    print(f"Target references: {len(refs)}")
    print(f"Physical scene groups: {len(groups)}")
    print(f"Separations: {sorted({ref.sep for ref in refs})}")
    print(f"Metrics: {args.metrics}")
    print(f"Output folder: {args.out_dir}")
    print("Loading IPDNet once...")
    ipd_model = load_ipdnet(args.ipd_ckpt, args.device)
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)
    ref_channel = int(getattr(dse_model, "ref_channel", 0))
    print(f"DSENet reference channel: {ref_channel}")
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    all_results: List[SeparationResult] = []
    skipped_no_doa = 0

    for (sep, fileid), scene_refs in tqdm(groups.items(), desc="DoA separation", unit="scene"):
        representative_mic = scene_refs[0].mic_path
        wav_tc, sr = load_multichannel_audio(representative_mic, target_sr=args.sample_rate)
        if ref_channel < 0 or ref_channel >= wav_tc.shape[1]:
            raise ValueError(
                f"Invalid DSENet ref_channel={ref_channel} for {representative_mic} with shape {wav_tc.shape}."
            )

        mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)
        noisy_ct = torch.from_numpy(wav_tc.T.copy())
        noisy_ref = wav_tc[:, ref_channel]
        duration_sec = len(noisy_ref) / float(sr)

        def run_ssl():
            with torch.inference_mode():
                return ipd_model(mic_batch)

        ssl_out, ipd_sec = elapsed_seconds(args.device, run_ssl)
        pred_doas = postprocess_doa_from_tensors(
            ssl_out["doa_est"],
            ssl_out["vad_est"],
            num_sources=args.num_sources,
            vad_th=args.vad_th,
            min_points_per_source=args.min_points_per_source,
        )
        candidate_pred_doas = ",".join(str(doa) for doa in pred_doas)

        if not pred_doas:
            skipped_no_doa += len(scene_refs)
            print(f"fileid={fileid}, sep={sep}: no usable IPDNet DOA, skipped {len(scene_refs)} targets.")
            continue

        target_items = []
        for ref in sorted(scene_refs, key=lambda item: item.speaker_id):
            pred_doa = nearest_predicted_doa(pred_doas, ref.gt_doa)
            if pred_doa is None:
                skipped_no_doa += 1
                continue
            target_items.append((ref, pred_doa))

        if not target_items:
            continue

        pred_doa_values = [pred_doa for _, pred_doa in target_items]
        dse_batch_size = len(pred_doa_values) if args.dse_batch_size <= 0 else args.dse_batch_size
        enhanced_items: List[np.ndarray] = []
        dse_time_by_index: List[float] = []

        for start in range(0, len(pred_doa_values), dse_batch_size):
            doa_chunk = pred_doa_values[start:start + dse_batch_size]
            enhanced_chunk, dse_chunk_sec = elapsed_seconds(
                args.device,
                lambda doa_chunk=doa_chunk: enhance_doa_batch(
                    dse_model,
                    noisy_ct,
                    doa_chunk,
                    args.width,
                    args.device,
                ),
            )
            enhanced_items.extend(enhanced_chunk)
            dse_time_by_index.extend([dse_chunk_sec / max(1, len(enhanced_chunk))] * len(enhanced_chunk))
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        for (ref, pred_doa), enhanced, dse_sec in zip(target_items, enhanced_items, dse_time_by_index):
            clean, clean_sr = load_mono_audio(ref.clean_path, target_sr=args.sample_rate)
            if clean_sr != sr:
                raise ValueError(f"Sample-rate mismatch: clean={clean_sr}, mic={sr} for {ref.clean_path}")

            audio_metrics = compute_audio_metrics(
                enhanced=enhanced,
                clean=clean,
                noisy_ref=noisy_ref,
                sample_rate=sr,
                metric_list=args.metrics,
            )

            enhanced_path = ""
            if args.save_enhanced:
                sep_dir = enhanced_root / f"sep{sep}"
                sep_dir.mkdir(parents=True, exist_ok=True)
                enhanced_name = (
                    f"enhanced_fileid_{fileid}_doa{ref.gt_doa}_sep{sep}_"
                    f"spk{ref.speaker_id}_preddoa{pred_doa}.wav"
                )
                enhanced_path = str(sep_dir / enhanced_name)
                sf.write(enhanced_path, enhanced, sr)

            reference_text = ref.text_path.read_text(encoding="utf-8").strip()

            def run_asr():
                return whisper_model.transcribe(
                    enhanced,
                    language=args.language,
                    fp16=args.whisper_device.startswith("cuda"),
                )

            asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
            hypothesis = asr_out.get("text", "").strip()
            sample_wer, dist, ref_words = wer(reference_text, hypothesis)
            doa_error = circular_angle_diff(pred_doa, ref.gt_doa)

            all_results.append(
                SeparationResult(
                    fileid=ref.fileid,
                    sep=ref.sep,
                    speaker_id=ref.speaker_id,
                    gt_doa=ref.gt_doa,
                    pred_doa=pred_doa,
                    doa_error_deg=doa_error,
                    candidate_pred_doas=candidate_pred_doas,
                    mic_file=ref.mic_path.name,
                    clean_file=ref.clean_path.name,
                    text_file=ref.text_path.name,
                    width=args.width,
                    duration_sec=duration_sec,
                    ref_channel=ref_channel,
                    input_sdr=audio_metrics["input_sdr"],
                    sdr=audio_metrics["sdr"],
                    sdr_i=audio_metrics["sdr_i"],
                    input_si_sdr=audio_metrics["input_si_sdr"],
                    si_sdr=audio_metrics["si_sdr"],
                    si_sdr_i=audio_metrics["si_sdr_i"],
                    input_pesq=audio_metrics["input_pesq"],
                    pesq=audio_metrics["pesq"],
                    pesq_i=audio_metrics["pesq_i"],
                    wer=sample_wer,
                    edit_distance=dist,
                    ref_words=ref_words,
                    ipdnet_sec=ipd_sec,
                    dsenet_sec=dse_sec,
                    whisper_sec=whisper_sec,
                    total_sec=ipd_sec + dse_sec + whisper_sec,
                    enhanced_wav=enhanced_path,
                    reference=reference_text,
                    hypothesis=hypothesis,
                )
            )

    detail_rows = [asdict(row) for row in all_results]
    details_csv = args.out_dir / "doa_separation_details.csv"
    write_csv(details_csv, detail_rows)

    summary_speaker_set = set(args.summary_speakers)
    summary_results = [row for row in all_results if row.speaker_id in summary_speaker_set]
    if not summary_results:
        raise RuntimeError(f"No evaluated rows matched --summary_speakers={args.summary_speakers}")

    summary_rows = summarize_rows(summary_results)
    summary_csv = args.out_dir / "doa_separation_summary_by_sep.csv"
    write_csv(summary_csv, summary_rows)

    speaker_summary_rows = summarize_rows_by_sep_speaker(all_results)
    speaker_summary_csv = args.out_dir / "doa_separation_summary_by_sep_speaker.csv"
    write_csv(speaker_summary_csv, speaker_summary_rows)

    summary_json = args.out_dir / "doa_separation_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "experiment": "IPDNet -> DSENet -> Whisper by true source DoA separation",
                "mic_dir": str(args.mic_dir),
                "clean_dir": str(args.clean_dir),
                "text_dir": str(args.text_dir),
                "ipd_ckpt": str(args.ipd_ckpt),
                "dse_ckpt": str(args.dse_ckpt),
                "whisper_model": args.whisper_model,
                "whisper_device": args.whisper_device,
                "device": args.device,
                "sample_rate": args.sample_rate,
                "width": args.width,
                "metrics": args.metrics,
                "target_references": len(refs),
                "physical_scene_groups": len(groups),
                "evaluated_items": len(all_results),
                "main_summary_speakers": args.summary_speakers,
                "main_summary_items": len(summary_results),
                "skipped": skipped,
                "skipped_no_doa": skipped_no_doa,
                "summary_by_sep": summary_rows,
                "summary_by_sep_speaker": speaker_summary_rows,
                "details_csv": str(details_csv),
                "summary_csv": str(summary_csv),
                "speaker_summary_csv": str(speaker_summary_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_paths = try_write_plots(args.out_dir, summary_rows)

    print(f"\n===== DOA SEPARATION SUMMARY (speakers {','.join(map(str, args.summary_speakers))}) =====")
    for row in summary_rows:
        def fmt(name: str, digits: int = 4) -> str:
            value = row.get(name)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        print(
            f"sep={row['sep']:>3} deg | "
            f"n={row['evaluated_items']} | "
            f"scenes={row['unique_scenes']} | "
            f"DOAerr={fmt('mean_doa_error_deg', 2)} deg | "
            f"WER={fmt('corpus_wer')} | "
            f"SI-SDRi={fmt('mean_sisdri')} | "
            f"SDRi={fmt('mean_sdri')} | "
            f"PESQ={fmt('mean_pesq')} | "
            f"PESQi={fmt('mean_pesqi')}"
        )

    print(f"\nSaved details: {details_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved speaker summary CSV: {speaker_summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
