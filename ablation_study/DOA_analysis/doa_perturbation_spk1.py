#!/usr/bin/env python3
"""
Forced-DOA perturbation experiment for DSENet + Whisper.

This script evaluates how a controlled DOA offset affects DSENet enhancement
and downstream ASR. It is based on offline/IPDNET/pipeline_full.py, but bypasses
IPDNet entirely:

    4-channel mixture + injected DOA=(GT_DOA + delta) % 360
        -> DSENet
        -> audio metrics against clean target
        -> Whisper
        -> WER

By default it evaluates the dominant speaker only, spk1, on the first 100
scenes. Nonzero perturbation magnitudes are randomly signed per scene:
    0, +/-5, +/-10, +/-20, +/-30 degrees
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
MODELS_ROOT = PROJECT_ROOT / "Models"
DSE_ROOT = MODELS_ROOT / "DSE"

sys.path.insert(0, str(DSE_ROOT))
from DOATrainer_3spk_myriad import TrainModule  # noqa: E402
from models.arch.DSENet import DSENet  # noqa: E402
from models.io.loss import Loss, MultiResolutionSTFTLoss  # noqa: E402
from models.io.norm import Norm  # noqa: E402
from models.io.stft import STFT  # noqa: E402
from models.utils.metrics import cal_metrics_functional, recover_scale  # noqa: E402


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


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.")
    return items


def parse_metric_list(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated metric list.")
    return items


def load_multichannel_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = np.stack(
            [resample_poly(wav[:, ch], up, down).astype(np.float32) for ch in range(wav.shape[1])],
            axis=1,
        )
        sr = target_sr
    return wav, sr


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


def find_single(candidates: Iterable[Path]) -> Optional[Path]:
    paths = sorted(candidates, key=lambda path: path.name)
    return paths[0] if paths else None


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    mic_path: Path
    clean_path: Path
    text_path: Path


def load_target_references(
    *,
    mic_dir: Path,
    clean_dir: Path,
    text_dir: Path,
    target_speaker_id: int,
    max_scenes: int,
    max_items: int,
) -> Tuple[List[TargetReference], Dict[str, int]]:
    refs: List[TargetReference] = []
    skipped = {
        "outside_max_scenes": 0,
        "non_target_speaker": 0,
        "missing_mic": 0,
        "missing_clean": 0,
    }

    text_paths = sorted(
        text_dir.glob("text_fileid_*_doa*_spk*.txt"),
        key=lambda p: (parse_fileid(p), parse_doa(p), p.name),
    )
    selected_fileids = sorted({parse_fileid(path) for path in text_paths})
    if max_scenes > 0:
        selected_fileids = selected_fileids[:max_scenes]
    selected_fileid_set = set(selected_fileids)

    for text_path in text_paths:
        fileid = parse_fileid(text_path)
        if fileid not in selected_fileid_set:
            skipped["outside_max_scenes"] += 1
            continue

        speaker_id = parse_speaker_id(text_path)
        if target_speaker_id != 0 and speaker_id != target_speaker_id:
            skipped["non_target_speaker"] += 1
            continue

        gt_doa = parse_doa(text_path)
        mic_path = find_single(mic_dir.glob(f"mic_fileid_{fileid}_doa{gt_doa}_*.wav"))
        if mic_path is None:
            skipped["missing_mic"] += 1
            continue

        clean_path = find_single(clean_dir.glob(f"clean_fileid_{fileid}_doa{gt_doa}_spk{speaker_id}.wav"))
        if clean_path is None:
            skipped["missing_clean"] += 1
            continue

        refs.append(
            TargetReference(
                fileid=fileid,
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


def load_dsenet(ckpt_path: Path, device: str) -> TrainModule:
    arch = DSENet(
        dim_input=8,
        dim_output=2,
        dim_squeeze=8,
        num_layers=8,
        num_freqs=129,
        encoder_kernel_size=5,
        dim_hidden=192,
        dim_ffn=192,
        num_heads=4,
        dropout=(0.0, 0.0, 0.0),
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        padding="zeros",
        full_share=0,
        d_embedding=40,
        d_alpha=20,
        width_emb_dim=3,
        width_stage=15,
        width_control=True,
    )
    stft = STFT(n_fft=256, n_hop=128, win_len=256, win="hann_window")
    norm = Norm(mode="frequency", online=True)
    loss = Loss(
        loss_func=MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512],
            hop_sizes=[120, 240, 50],
            win_lengths=[600, 1200, 240],
            window="hann_window",
        ),
        pit=False,
        loss_func_kwargs={},
    )

    model = TrainModule.load_from_checkpoint(
        ckpt_path,
        arch=arch,
        stft=stft,
        norm=norm,
        loss=loss,
        map_location=device,
    )
    model.eval().to(device).float()
    return model


def enhance_doa_batch(
    dse_model: TrainModule,
    noisy_ct: torch.Tensor,
    doa_values: Sequence[int],
    width_value: int,
    device: str,
) -> List[np.ndarray]:
    batch_size = len(doa_values)
    if batch_size == 0:
        return []

    x = noisy_ct.unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)
    doa = torch.tensor(doa_values, dtype=torch.long, device=device)
    width = torch.full((batch_size,), width_value, dtype=torch.long, device=device)

    with torch.inference_mode():
        yr_hat = dse_model.forward(x, doa, width)
        if dse_model.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(
                preds=yr_hat,
                mixture=x[:, dse_model.ref_channel, :],
                scale_src_together=True,
                norm_if_exceed_1=False,
            )

    return [yr_hat[idx, 0].detach().cpu().numpy().astype(np.float32) for idx in range(batch_size)]


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


@dataclass
class PerturbationResult:
    fileid: int
    speaker_id: int
    mic_file: str
    clean_file: str
    text_file: str
    gt_doa: int
    delta_magnitude_deg: int
    signed_delta_deg: int
    injected_doa: int
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
    dsenet_sec: float
    whisper_sec: float
    total_sec: float
    enhanced_wav: str
    reference: str
    hypothesis: str


def mean_optional(rows: Sequence[PerturbationResult], attr: str) -> Optional[float]:
    values = [
        float(value)
        for row in rows
        for value in [getattr(row, attr)]
        if value is not None and np.isfinite(float(value))
    ]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: Sequence[PerturbationResult]) -> List[Dict[str, Optional[float] | int]]:
    summary_rows: List[Dict[str, Optional[float] | int]] = []
    for delta_magnitude in sorted({row.delta_magnitude_deg for row in rows}):
        delta_rows = [row for row in rows if row.delta_magnitude_deg == delta_magnitude]
        total_edits = sum(row.edit_distance for row in delta_rows)
        total_ref_words = sum(row.ref_words for row in delta_rows)
        summary_rows.append(
            {
                "delta_magnitude_deg": delta_magnitude,
                "evaluated_items": len(delta_rows),
                "negative_delta_count": sum(1 for row in delta_rows if row.signed_delta_deg < 0),
                "positive_delta_count": sum(1 for row in delta_rows if row.signed_delta_deg > 0),
                "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
                "mean_sample_wer": float(np.mean([row.wer for row in delta_rows])) if delta_rows else 0.0,
                "mean_input_sdr": mean_optional(delta_rows, "input_sdr"),
                "mean_sdr": mean_optional(delta_rows, "sdr"),
                "mean_sdri": mean_optional(delta_rows, "sdr_i"),
                "mean_input_si_sdr": mean_optional(delta_rows, "input_si_sdr"),
                "mean_si_sdr": mean_optional(delta_rows, "si_sdr"),
                "mean_sisdri": mean_optional(delta_rows, "si_sdr_i"),
                "mean_input_pesq": mean_optional(delta_rows, "input_pesq"),
                "mean_pesq": mean_optional(delta_rows, "pesq"),
                "mean_pesqi": mean_optional(delta_rows, "pesq_i"),
                "mean_dsenet_sec": mean_optional(delta_rows, "dsenet_sec"),
                "mean_whisper_sec": mean_optional(delta_rows, "whisper_sec"),
                "mean_total_sec": mean_optional(delta_rows, "total_sec"),
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
        for row in rows:
            writer.writerow(row)


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
    deltas = [int(row["delta_magnitude_deg"]) for row in summary_rows]
    written: List[str] = []

    def values(name: str) -> List[float]:
        return [float(row[name]) if row.get(name) is not None else np.nan for row in summary_rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax.plot(deltas, values("corpus_wer"), marker="o", label="Corpus WER")
    ax.plot(deltas, values("mean_sample_wer"), marker="s", label="Mean sample WER")
    ax.set_xlabel("Injected DOA offset magnitude (deg)")
    ax.set_ylabel("WER")
    ax.set_title("Whisper WER vs DSENet DOA Offset")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    wer_plot = plot_dir / "wer_vs_delta.png"
    fig.savefig(wer_plot)
    plt.close(fig)
    written.append(str(wer_plot))

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    ax1.plot(deltas, values("mean_sdri"), marker="o", label="SDRi")
    ax1.plot(deltas, values("mean_sisdri"), marker="s", label="SI-SDRi")
    ax1.set_xlabel("Injected DOA offset magnitude (deg)")
    ax1.set_ylabel("Improvement (dB)")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(deltas, values("mean_pesqi"), marker="^", color="#54a24b", label="PESQi")
    ax2.set_ylabel("PESQ improvement")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False)
    ax1.set_title("Enhancement Metrics vs DSENet DOA Offset")
    fig.tight_layout()
    metric_plot = plot_dir / "enhancement_metrics_vs_delta.png"
    fig.savefig(metric_plot)
    plt.close(fig)
    written.append(str(metric_plot))

    return written


def build_signed_delta_items(
    *,
    fileid: int,
    gt_doa: int,
    delta_magnitudes: Sequence[int],
    rng: random.Random,
    fixed_positive_deltas: bool,
    sign_cache: Dict[Tuple[int, int], int],
) -> List[Tuple[int, int, int]]:
    items: List[Tuple[int, int, int]] = []
    for magnitude in delta_magnitudes:
        magnitude = abs(int(magnitude))
        if magnitude == 0:
            signed_delta = 0
        elif fixed_positive_deltas:
            signed_delta = magnitude
        else:
            cache_key = (fileid, magnitude)
            if cache_key not in sign_cache:
                sign_cache[cache_key] = -1 if rng.random() < 0.5 else 1
            signed_delta = sign_cache[cache_key] * magnitude
        injected_doa = (gt_doa + signed_delta) % 360
        items.append((magnitude, signed_delta, injected_doa))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DSENet sensitivity to forced GT_DOA +/- delta inputs for spk1."
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_v13_99.ckpt")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "dsenet_spk1_gt_doa_delta",
    )
    parser.add_argument("--target_speaker_id", type=int, default=1, help="Use 1 for spk1, or 0 for all speakers.")
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=100,
        help="Use only the first N fileid scenes after numeric sorting; 0 means all scenes.",
    )
    parser.add_argument(
        "--deltas",
        type=parse_int_list,
        default=parse_int_list("0,5,10,15,20,30"),
        help=(
            "Comma-separated DOA perturbation magnitudes. Nonzero values are randomly signed "
            "per scene unless --fixed_positive_deltas is set."
        ),
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Seed for the +/- perturbation signs.")
    parser.add_argument(
        "--fixed_positive_deltas",
        action="store_true",
        help="Disable random signs and use +delta for all nonzero perturbations.",
    )
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument(
        "--metrics",
        type=parse_metric_list,
        default=parse_metric_list("SDR,SI_SDR,WB_PESQ"),
        help="Comma-separated metrics passed to cal_metrics_functional.",
    )
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--dse_batch_size",
        type=int,
        default=0,
        help="How many delta DOAs to enhance per DSENet pass. 0 means all deltas for each target.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Optional extra cap on target references; 0 means no cap.")
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
    if not args.dse_ckpt.is_file():
        raise FileNotFoundError(f"DSENet checkpoint not found: {args.dse_ckpt}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DSENet was requested on CUDA, but torch.cuda.is_available() is False.")
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_root = args.out_dir / "enhanced"
    if args.save_enhanced:
        enhanced_root.mkdir(parents=True, exist_ok=True)

    refs, skipped = load_target_references(
        mic_dir=args.mic_dir,
        clean_dir=args.clean_dir,
        text_dir=args.text_dir,
        target_speaker_id=args.target_speaker_id,
        max_scenes=args.max_scenes,
        max_items=args.max_items,
    )
    if not refs:
        raise RuntimeError("No target references were found. Check target speaker id and dataset paths.")

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"Target speaker id: {args.target_speaker_id}")
    print(f"Max scenes: {args.max_scenes if args.max_scenes > 0 else 'all'}")
    print(f"Target references: {len(refs)}")
    print(f"Delta magnitudes: {args.deltas}")
    print(f"Random signed deltas: {not args.fixed_positive_deltas}")
    print(f"Random seed: {args.random_seed}")
    print(f"Width: {args.width}")
    print(f"Metrics: {args.metrics}")
    print(f"Output folder: {args.out_dir}")
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)
    ref_channel = int(getattr(dse_model, "ref_channel", 0))
    print(f"DSENet reference channel: {ref_channel}")
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    all_results: List[PerturbationResult] = []
    rng = random.Random(args.random_seed)
    sign_cache: Dict[Tuple[int, int], int] = {}

    for ref in tqdm(refs, desc="DOA perturbation", unit="target"):
        wav_tc, sr = load_multichannel_audio(ref.mic_path, target_sr=args.sample_rate)
        if ref_channel < 0 or ref_channel >= wav_tc.shape[1]:
            raise ValueError(f"Invalid DSENet ref_channel={ref_channel} for {ref.mic_path} with shape {wav_tc.shape}.")

        noisy_ct = torch.from_numpy(wav_tc.T.copy())
        noisy_ref = wav_tc[:, ref_channel]
        clean, clean_sr = load_mono_audio(ref.clean_path, target_sr=args.sample_rate)
        if clean_sr != sr:
            raise ValueError(f"Unexpected sample-rate mismatch for {ref.clean_path}: clean={clean_sr}, mic={sr}.")
        reference_text = ref.text_path.read_text(encoding="utf-8").strip()
        duration_sec = len(noisy_ref) / float(sr)

        delta_items = build_signed_delta_items(
            fileid=ref.fileid,
            gt_doa=ref.gt_doa,
            delta_magnitudes=args.deltas,
            rng=rng,
            fixed_positive_deltas=args.fixed_positive_deltas,
            sign_cache=sign_cache,
        )
        dse_batch_size = len(delta_items) if args.dse_batch_size <= 0 else args.dse_batch_size

        for start in range(0, len(delta_items), dse_batch_size):
            delta_chunk = delta_items[start:start + dse_batch_size]
            doa_chunk = [injected_doa for _, _, injected_doa in delta_chunk]
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
            dse_sec_each = dse_chunk_sec / max(1, len(enhanced_chunk))

            for (delta_magnitude, signed_delta, injected_doa), enhanced in zip(delta_chunk, enhanced_chunk):
                enhanced_path = ""
                if args.save_enhanced:
                    delta_dir = enhanced_root / f"delta_{delta_magnitude:03d}"
                    delta_dir.mkdir(parents=True, exist_ok=True)
                    enhanced_file = (
                        f"enhanced_fileid_{ref.fileid}_spk{ref.speaker_id}_"
                        f"gt{ref.gt_doa}_signed_delta{signed_delta:+d}_doa{injected_doa}.wav"
                    )
                    enhanced_path = str(delta_dir / enhanced_file)
                    sf.write(enhanced_path, enhanced, sr)

                audio_metrics = compute_audio_metrics(
                    enhanced=enhanced,
                    clean=clean,
                    noisy_ref=noisy_ref,
                    sample_rate=sr,
                    metric_list=args.metrics,
                )

                def run_asr():
                    return whisper_model.transcribe(
                        enhanced,
                        language=args.language,
                        fp16=args.whisper_device.startswith("cuda"),
                    )

                asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
                hypothesis = asr_out.get("text", "").strip()
                sample_wer, dist, ref_words = wer(reference_text, hypothesis)

                all_results.append(
                    PerturbationResult(
                        fileid=ref.fileid,
                        speaker_id=ref.speaker_id,
                        mic_file=str(ref.mic_path),
                        clean_file=str(ref.clean_path),
                        text_file=str(ref.text_path),
                        gt_doa=ref.gt_doa,
                        delta_magnitude_deg=delta_magnitude,
                        signed_delta_deg=signed_delta,
                        injected_doa=injected_doa,
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
                        dsenet_sec=dse_sec_each,
                        whisper_sec=whisper_sec,
                        total_sec=dse_sec_each + whisper_sec,
                        enhanced_wav=enhanced_path,
                        reference=reference_text,
                        hypothesis=hypothesis,
                    )
                )

            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    details_csv = args.out_dir / "doa_perturbation_wer_details.csv"
    detail_rows = [asdict(row) for row in all_results]
    write_csv(details_csv, detail_rows)

    summary_rows = summarize_rows(all_results)
    summary_csv = args.out_dir / "doa_perturbation_wer_summary_by_delta.csv"
    write_csv(summary_csv, summary_rows)

    summary_json = args.out_dir / "doa_perturbation_wer_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "experiment": "DSENet forced GT_DOA + delta perturbation",
                "mic_dir": str(args.mic_dir),
                "clean_dir": str(args.clean_dir),
                "text_dir": str(args.text_dir),
                "dse_ckpt": str(args.dse_ckpt),
                "target_speaker_id": args.target_speaker_id,
                "max_scenes": args.max_scenes,
                "delta_magnitudes": args.deltas,
                "random_signed_deltas": not args.fixed_positive_deltas,
                "random_seed": args.random_seed,
                "width": args.width,
                "metrics": args.metrics,
                "whisper_model": args.whisper_model,
                "whisper_device": args.whisper_device,
                "device": args.device,
                "sample_rate": args.sample_rate,
                "target_references": len(refs),
                "evaluated_items": len(all_results),
                "skipped": skipped,
                "summary_by_delta": summary_rows,
                "details_csv": str(details_csv),
                "summary_csv": str(summary_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_paths = try_write_plots(args.out_dir, summary_rows)

    print("\n===== DSENET DOA PERTURBATION SUMMARY =====")
    for row in summary_rows:
        def fmt(name: str, digits: int = 4) -> str:
            value = row.get(name)
            return "N/A" if value is None else f"{float(value):.{digits}f}"

        print(
            f"delta_mag={row['delta_magnitude_deg']:>3} deg | "
            f"n={row['evaluated_items']} | "
            f"-/+={row['negative_delta_count']}/{row['positive_delta_count']} | "
            f"WER={fmt('corpus_wer')} | "
            f"SI-SDRi={fmt('mean_sisdri')} | "
            f"SDRi={fmt('mean_sdri')} | "
            f"PESQ={fmt('mean_pesq')}"
        )

    print(f"\nSaved details: {details_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
