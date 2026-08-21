"""
Offline GT-DoA -> DSE -> ASR pipeline.

Loads DSENet and Whisper once, then processes saved multi-channel wavs for the
target speaker. DSENet receives the ground-truth DoA parsed from each target
filename. The script also computes SDR/SI-SDR/WB-PESQ against the clean target,
using one noisy mixture channel as the input baseline.
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


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
DSE_ROOT = MODELS_ROOT / "DSE"

sys.path.insert(0, str(DSE_ROOT))
from DOATrainer_3spk_myriad import TrainModule  # noqa: E402
from models.arch.DSENet import DSENet  # noqa: E402
from models.utils.metrics import cal_metrics_functional, recover_scale  # noqa: E402
from models.io.loss import Loss, MultiResolutionSTFTLoss
from models.io.norm import Norm
from models.io.stft import STFT


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_seconds(device: str, fn):
    cuda_sync(device)
    start = time.perf_counter()
    value = fn()
    cuda_sync(device)
    return value, time.perf_counter() - start


def circular_angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean_deg_360(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    angles_deg = np.asarray(angles_deg) % 360.0
    angles_rad = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(angles_rad)
    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


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


def parse_spk(path_or_name: Path | str) -> str:
    match = re.search(r"spk(\d+)", Path(path_or_name).stem)
    return match.group(1) if match else ""


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


def mean_or_zero(values: Iterable[Optional[float]]) -> float:
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else 0.0


def load_multichannel_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)  # [T, C]
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


def make_text_name_from_mic(mic_path: Path, text_dir: Path) -> Path:
    fileid = parse_fileid(mic_path)
    doa = parse_doa(mic_path)
    candidates = sorted(text_dir.glob(f"text_fileid_{fileid}_doa{doa}_spk*.txt"))
    if candidates:
        return candidates[0]
    return text_dir / f"text_fileid_{fileid}_doa{doa}.txt"


def make_clean_name_from_target(target_path: Path, clean_dir: Path, speaker_id: str) -> Path:
    fileid = parse_fileid(target_path)
    doa = parse_doa(target_path)
    return clean_dir / f"clean_fileid_{fileid}_doa{doa}_spk{speaker_id}.wav"


def unique_mic_files(mic_dir: Path, max_items: int) -> List[Path]:
    all_files = sorted(mic_dir.glob("*.wav"), key=lambda p: (parse_fileid(p), parse_doa(p), p.name))
    if max_items > 0:
        return all_files[:max_items]
    return all_files


def group_targets_by_fileid(mic_files: Iterable[Path]) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in mic_files:
        grouped.setdefault(parse_fileid(path), []).append(path)
    return grouped


def choose_representative_mic(target_paths: Sequence[Path]) -> Path:
    return sorted(target_paths, key=lambda p: (parse_doa(p), p.name))[0]


def postprocess_doa_from_tensors(
    doa_est: torch.Tensor,
    vad_est: torch.Tensor,
    num_sources: int,
    vad_th: float,
    min_points_per_source: int,
) -> List[int]:
    doa_np = doa_est.detach().cpu().numpy()
    vad_np = vad_est.detach().cpu().numpy()

    azi = doa_np[0, :, 1, :] % 360.0
    score = vad_np[0, :, :]
    active = score > vad_th

    valid_angles = []
    valid_weights = []
    for t in range(azi.shape[0]):
        for k in range(azi.shape[1]):
            if active[t, k]:
                valid_angles.append(azi[t, k])
                valid_weights.append(max(float(score[t, k]), 1e-6))

    if len(valid_angles) < num_sources:
        return []

    valid_angles_np = np.asarray(valid_angles, dtype=np.float32)
    valid_weights_np = np.asarray(valid_weights, dtype=np.float32)
    angle_rad = np.deg2rad(valid_angles_np)
    xy = np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=1)

    n_clusters = min(num_sources, len(valid_angles_np))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(
        xy,
        sample_weight=valid_weights_np,
    )

    final_doas: List[int] = []
    for source_id in range(n_clusters):
        cluster_angles = valid_angles_np[labels == source_id]
        cluster_weights = valid_weights_np[labels == source_id]
        if len(cluster_angles) < min_points_per_source:
            continue
        final_doas.append(int(round(circular_mean_deg_360(cluster_angles, cluster_weights))) % 360)

    return sorted(set(final_doas))


def nearest_predicted_doa(pred_doas: Sequence[int], target_doa: int) -> Optional[int]:
    if not pred_doas:
        return None
    return int(min(pred_doas, key=lambda pred: circular_angle_diff(pred, target_doa)))


def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


RESPEAKER4_RADIUS_M = 0.031
RESPEAKER4_MIC_POS = np.array(
    (
        (RESPEAKER4_RADIUS_M / np.sqrt(2), RESPEAKER4_RADIUS_M / np.sqrt(2), 0.0),
        (-RESPEAKER4_RADIUS_M / np.sqrt(2), RESPEAKER4_RADIUS_M / np.sqrt(2), 0.0),
        (-RESPEAKER4_RADIUS_M / np.sqrt(2), -RESPEAKER4_RADIUS_M / np.sqrt(2), 0.0),
        (RESPEAKER4_RADIUS_M / np.sqrt(2), -RESPEAKER4_RADIUS_M / np.sqrt(2), 0.0),
    ),
    dtype=np.float32,
)


class IPDNetInference(torch.nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device_name = device
        self.arch = ssl_model.IPDnet(
            input_size=8,
            hidden_size=128,
            max_track=3,
            is_online=True,
        )
        self.dostft = ssl_module.STFT(win_len=512, win_shift_ratio=0.5, nfft=512)
        self.fre_range_used = range(1, 257)
        self.doa_decoder = ssl_module.PredDOA(
            mic_location=RESPEAKER4_MIC_POS,
            max_track=3,
            max_num_sources=1,
            res_phi=360,
            dev=device,
            is_linear_array=False,
            is_planar_array=True,
        )

    def forward(self, mic_sig_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        in_batch = self.data_preprocess_inference(mic_sig_batch)[0]
        pred_ipd = self.arch(in_batch)
        doa_est, vad_est = self.pred_ipd_to_doa(pred_ipd)
        return {"doa_est": doa_est, "vad_est": vad_est, "pred_ipd": pred_ipd}

    def data_preprocess_inference(self, mic_sig_batch: torch.Tensor, eps: float = 1e-6) -> List[torch.Tensor]:
        stft = self.dostft(signal=mic_sig_batch)
        stft_rebatch = stft.permute(0, 3, 1, 2).to(self.device_name)
        mag = torch.abs(stft_rebatch)
        mean_value = forgetting_norm(mag, sample_length=280)
        stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
        stft_rebatch_imag = torch.imag(stft_rebatch) / (mean_value + eps)
        real_imag_batch = torch.cat((stft_rebatch_real, stft_rebatch_imag), dim=1)
        return [real_imag_batch[:, :, self.fre_range_used, :]]

    def pred_ipd_to_doa(self, pred_ipd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nb, nt, ndoa, nmic, ntrack = pred_ipd.shape
        pred_ipd_tracks = pred_ipd.permute(0, 3, 1, 2, 4).reshape(nb * nmic, nt, ndoa, ntrack)

        doa_tracks = []
        vad_tracks = []
        for track_idx in range(ntrack):
            pred_track, _ = self.doa_decoder.pred2DOA_track(
                pred_batch=pred_ipd_tracks[:, :, :, track_idx],
                gt_batch=None,
            )
            doa_tracks.append(pred_track[0])
            vad_tracks.append(pred_track[1])

        doa_est = torch.cat(doa_tracks, dim=-1) * 180.0 / np.pi
        vad_est = torch.cat(vad_tracks, dim=-1)
        return doa_est, vad_est


def load_ipdnet(ckpt_path: Path, device: str) -> IPDNetInference:
    model = IPDNetInference(device=device)
    ckpt = torch_load_checkpoint(ckpt_path, device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    arch_state = {}
    for key, value in state_dict.items():
        if key.startswith("arch."):
            arch_state[key.replace("arch.", "", 1)] = value
        elif key.startswith("model.arch."):
            arch_state[key.replace("model.arch.", "", 1)] = value
        else:
            arch_state[key] = value

    model_keys = model.arch.state_dict()
    arch_state = {
        key: value
        for key, value in arch_state.items()
        if key in model_keys and hasattr(value, "shape") and tuple(model_keys[key].shape) == tuple(value.shape)
    }
    if not arch_state:
        raise RuntimeError(f"No IPDNET arch weights matched checkpoint: {ckpt_path}")

    missing, unexpected = model.arch.load_state_dict(arch_state, strict=False)
    if unexpected:
        print(f"Warning: IPDNET ignored {len(unexpected)} unexpected checkpoint keys.")
    if missing:
        print(f"Warning: IPDNET missing {len(missing)} arch keys while loading checkpoint.")

    model.eval().to(device)
    model.doa_decoder.to(device)
    return model


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

    print("Loading checkpoint...")
    model = TrainModule.load_from_checkpoint(
        ckpt_path,
        arch=arch,
        stft=stft,
        norm=norm,
        loss=loss,
        map_location=device
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

    x = noisy_ct.unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)  # [B, C, T]
    doa = torch.tensor(doa_values, dtype=torch.long, device=device)  # [B]
    width = torch.full((batch_size,), width_value, dtype=torch.long, device=device)  # [B]

    with torch.inference_mode():
        yr_hat = dse_model.forward(x, doa, width)
        if dse_model.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(
                preds=yr_hat,
                mixture=x[:, dse_model.ref_channel, :],
                scale_src_together=True,
                norm_if_exceed_1=False,
            )

    return [
        yr_hat[idx, 0].detach().cpu().numpy().astype(np.float32)
        for idx in range(batch_size)
    ]


@dataclass
class TargetResult:
    fileid: int
    mic_file: str
    text_file: str
    gt_doa: int
    pred_doa: Optional[int]
    doa_error_deg: Optional[float]
    spk: str
    clean_file: str
    wer: float
    edit_distance: int
    ref_words: int
    input_sdr: Optional[float]
    sdr: Optional[float]
    sdr_i: Optional[float]
    input_si_sdr: Optional[float]
    si_sdr: Optional[float]
    si_sdr_i: Optional[float]
    input_wb_pesq: Optional[float]
    wb_pesq: Optional[float]
    wb_pesq_i: Optional[float]
    dsenet_sec: float
    metrics_sec: float
    whisper_sec: float
    total_target_sec: float
    reference: str
    hypothesis: str


def summarize_by_speaker(results: Sequence[TargetResult]) -> Dict[str, Dict[str, float | int]]:
    summary: Dict[str, Dict[str, float | int]] = {}
    for spk in sorted({r.spk for r in results if r.spk}):
        spk_results = [r for r in results if r.spk == spk]
        total_edits = sum(r.edit_distance for r in spk_results)
        total_ref_words = sum(r.ref_words for r in spk_results)
        doa_errors = [
            float(r.doa_error_deg)
            for r in spk_results
            if r.doa_error_deg is not None
        ]
        spk_summary: Dict[str, float | int] = {
            "items": len(spk_results),
            "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
            "mean_sample_wer": float(np.mean([r.wer for r in spk_results])) if spk_results else 0.0,
            "mean_doa_error_deg": float(np.mean(doa_errors)) if doa_errors else 0.0,
            "median_doa_error_deg": float(np.median(doa_errors)) if doa_errors else 0.0,
            "mean_input_sdr": mean_or_zero(r.input_sdr for r in spk_results),
            "mean_sdr": mean_or_zero(r.sdr for r in spk_results),
            "mean_sdri": mean_or_zero(r.sdr_i for r in spk_results),
            "mean_input_si_sdr": mean_or_zero(r.input_si_sdr for r in spk_results),
            "mean_si_sdr": mean_or_zero(r.si_sdr for r in spk_results),
            "mean_sisdri": mean_or_zero(r.si_sdr_i for r in spk_results),
            "mean_input_wb_pesq": mean_or_zero(r.input_wb_pesq for r in spk_results),
            "mean_wb_pesq": mean_or_zero(r.wb_pesq for r in spk_results),
            "mean_wb_pesqi": mean_or_zero(r.wb_pesq_i for r in spk_results),
        }
        for threshold in (10, 20, 30):
            within = sum(1 for err in doa_errors if err <= threshold)
            spk_summary[f"doa_within_{threshold}_deg_count"] = within
            spk_summary[f"doa_within_{threshold}_deg_percent"] = (
                100.0 * within / len(doa_errors)
                if doa_errors
                else 0.0
            )
        summary[f"spk{spk}"] = spk_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT-DoA -> DSENet -> Whisper in one process.")
    parser.add_argument("--mic_dir", type=Path, default=PROJECT_ROOT / "data" / "dataset_4mic_3spk" / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=PROJECT_ROOT / "data" / "dataset_4mic_3spk" / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=PROJECT_ROOT / "data" / "dataset_4mic_3spk" / "Eval" / "text")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_v13_99.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--target_speaker_id", type=int, default=0, help="Use 0 for all speakers, or 1 for dominant spk1 only.")
    parser.add_argument("--mic_channel", type=int, default=1, help="1-based mixture channel used as noisy baseline.")
    parser.add_argument("--metric_list", type=str, default="SDR,SI_SDR,WB_PESQ")
    parser.add_argument(
        "--dse_batch_size",
        type=int,
        default=1,
        help="How many DOAs to enhance per DSENet forward pass. Use 3 for speed, 1 for lower VRAM.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit target wav files for a quick test; 0 means all.")
    parser.add_argument(
        "--save_enhanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save enhanced GT-DoA wavs. Use --no-save_enhanced to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    metric_list = [item.strip() for item in args.metric_list.split(",") if item.strip()]
    if not metric_list:
        raise ValueError("--metric_list must contain at least one metric.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_enhanced_GT"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"DSENet batch size: {args.dse_batch_size}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Clean target folder: {args.clean_dir}")
    print(f"Ground-truth text folder: {args.text_dir}")
    print(f"Target speaker: {'all' if args.target_speaker_id == 0 else f'spk{args.target_speaker_id}'}")
    print("Using ground-truth DoA from target filenames; IPDNet is not run.")
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print("Note: this script does not cut audio into chunks; it processes the saved wav files as-is.")

    all_results: List[TargetResult] = []
    total_edits = 0
    total_ref_words = 0
    missing_text = 0
    skipped_no_doa = 0
    skipped_non_target_speaker = 0
    missing_clean = 0
    metric_failures = 0

    for fileid, target_paths in tqdm(grouped.items(), desc="GT-DoA pipeline", unit="scene"):
        mic_path = choose_representative_mic(target_paths)
        wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
        noisy_ct = torch.from_numpy(wav_tc.T.copy())
        mic_idx = args.mic_channel - 1
        if mic_idx < 0 or mic_idx >= wav_tc.shape[1]:
            raise ValueError(f"--mic_channel {args.mic_channel} is invalid for {wav_tc.shape[1]}-channel file: {mic_path}")
        noisy_ref = wav_tc[:, mic_idx]

        valid_targets = []
        for target_path in target_paths:
            text_path = make_text_name_from_mic(target_path, args.text_dir)
            if not text_path.is_file():
                missing_text += 1
                print(f"fileid={fileid}: missing text {text_path.name}")
                continue

            spk = parse_spk(text_path)
            if args.target_speaker_id != 0 and spk != str(args.target_speaker_id):
                skipped_non_target_speaker += 1
                continue

            clean_path = make_clean_name_from_target(target_path, args.clean_dir, spk)
            if not clean_path.is_file():
                missing_clean += 1
                print(f"fileid={fileid}: missing clean target {clean_path.name}")
                continue

            gt_doa = parse_doa(target_path)
            pred_doa = gt_doa

            valid_targets.append((target_path, text_path, clean_path, gt_doa, pred_doa, spk))

        if not valid_targets:
            continue

        batched_pred_doas = [item[3] for item in valid_targets]
        dse_batch_size = len(batched_pred_doas) if args.dse_batch_size <= 0 else args.dse_batch_size
        enhanced_batch = []
        dse_sec = 0.0

        for start in range(0, len(batched_pred_doas), dse_batch_size):
            doa_chunk = batched_pred_doas[start:start + dse_batch_size]
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
            enhanced_batch.extend(enhanced_chunk)
            dse_sec += dse_chunk_sec
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        for (target_path, text_path, clean_path, gt_doa, pred_doa, spk), enhanced in zip(valid_targets, enhanced_batch):
            doa_error = circular_angle_diff(pred_doa, gt_doa)

            if args.save_enhanced:
                save_name = f"enhanced_fileid_{fileid}_gt{gt_doa}_pred{pred_doa}_spk{spk}.wav"
                sf.write(str(enhanced_dir / save_name), enhanced, sr)

            clean_audio, _ = load_mono_audio(clean_path, target_sr=args.sample_rate)
            metrics, metrics_sec = elapsed_seconds(
                args.device,
                lambda enhanced=enhanced, clean_audio=clean_audio, noisy_ref=noisy_ref: compute_audio_metrics(
                    enhanced=enhanced,
                    clean=clean_audio,
                    noisy_ref=noisy_ref,
                    sample_rate=sr,
                    metric_list=metric_list,
                ),
            )
            if not any(value is not None for value in metrics.values()):
                metric_failures += 1

            def run_asr():
                return whisper_model.transcribe(
                    enhanced,
                    language=args.language,
                    fp16=args.whisper_device.startswith("cuda"),
                )

            asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
            hyp_text = asr_out.get("text", "").strip()
            ref_text = text_path.read_text(encoding="utf-8").strip()
            sample_wer, dist, ref_words = wer(ref_text, hyp_text)

            total_edits += dist
            total_ref_words += ref_words

            all_results.append(
                TargetResult(
                    fileid=fileid,
                    mic_file=target_path.name,
                    text_file=text_path.name,
                    gt_doa=gt_doa,
                    pred_doa=pred_doa,
                    doa_error_deg=doa_error,
                    spk=spk,
                    clean_file=clean_path.name,
                    wer=sample_wer,
                    edit_distance=dist,
                    ref_words=ref_words,
                    input_sdr=metrics["input_sdr"],
                    sdr=metrics["sdr"],
                    sdr_i=metrics["sdr_i"],
                    input_si_sdr=metrics["input_si_sdr"],
                    si_sdr=metrics["si_sdr"],
                    si_sdr_i=metrics["si_sdr_i"],
                    input_wb_pesq=metrics["input_wb_pesq"],
                    wb_pesq=metrics["wb_pesq"],
                    wb_pesq_i=metrics["wb_pesq_i"],
                    dsenet_sec=dse_sec,
                    metrics_sec=metrics_sec,
                    whisper_sec=whisper_sec,
                    total_target_sec=dse_sec + metrics_sec + whisper_sec,
                    reference=ref_text,
                    hypothesis=hyp_text,
                )
            )

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_gt_doa_wer_details.csv"
    fieldnames = list(TargetResult.__dataclass_fields__.keys())
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(asdict(row))

    corpus_wer = (total_edits / total_ref_words) if total_ref_words > 0 else 0.0
    mean_sample_wer = float(np.mean([r.wer for r in all_results])) if all_results else 0.0
    by_speaker = summarize_by_speaker(all_results)

    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
        "text_dir": str(args.text_dir),
        "dse_ckpt": str(args.dse_ckpt),
        "doa_source": "ground_truth",
        "target_speaker_id": args.target_speaker_id,
        "mic_channel": args.mic_channel,
        "metric_list": metric_list,
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "target_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "evaluated_items": len(all_results),
        "missing_text": missing_text,
        "missing_clean": missing_clean,
        "skipped_non_target_speaker": skipped_non_target_speaker,
        "skipped_no_doa": skipped_no_doa,
        "metric_failures": metric_failures,
        "corpus_wer": corpus_wer,
        "mean_sample_wer": mean_sample_wer,
        "by_speaker": by_speaker,
        "mean_input_sdr": mean_or_zero(r.input_sdr for r in all_results),
        "mean_sdr": mean_or_zero(r.sdr for r in all_results),
        "mean_sdri": mean_or_zero(r.sdr_i for r in all_results),
        "mean_input_si_sdr": mean_or_zero(r.input_si_sdr for r in all_results),
        "mean_si_sdr": mean_or_zero(r.si_sdr for r in all_results),
        "mean_sisdri": mean_or_zero(r.si_sdr_i for r in all_results),
        "mean_input_wb_pesq": mean_or_zero(r.input_wb_pesq for r in all_results),
        "mean_wb_pesq": mean_or_zero(r.wb_pesq for r in all_results),
        "mean_wb_pesqi": mean_or_zero(r.wb_pesq_i for r in all_results),
        "mean_dsenet_sec": float(np.mean([r.dsenet_sec for r in all_results])) if all_results else 0.0,
        "mean_metrics_sec": float(np.mean([r.metrics_sec for r in all_results])) if all_results else 0.0,
        "mean_whisper_sec": float(np.mean([r.whisper_sec for r in all_results])) if all_results else 0.0,
        "mean_total_target_sec": float(np.mean([r.total_target_sec for r in all_results])) if all_results else 0.0,
    }

    speaker_detail_csvs: Dict[str, str] = {}
    speaker_summary_jsons: Dict[str, str] = {}
    for spk in sorted({row.spk for row in all_results if row.spk}, key=lambda value: int(value)):
        spk_rows = [row for row in all_results if row.spk == spk]
        spk_details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_gt_doa_wer_details_spk{spk}.csv"
        with spk_details_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in spk_rows:
                writer.writerow(asdict(row))

        spk_summary = {
            "doa_source": "ground_truth",
            "speaker": f"spk{spk}",
            "evaluated_items": len(spk_rows),
            **by_speaker[f"spk{spk}"],
            "details_csv": str(spk_details_csv),
        }
        spk_summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_gt_doa_wer_summary_spk{spk}.json"
        spk_summary_json.write_text(json.dumps(spk_summary, indent=2), encoding="utf-8")
        speaker_detail_csvs[f"spk{spk}"] = str(spk_details_csv)
        speaker_summary_jsons[f"spk{spk}"] = str(spk_summary_json)

    summary["details_csv"] = str(details_csv)
    summary["speaker_detail_csvs"] = speaker_detail_csvs
    summary["speaker_summary_jsons"] = speaker_summary_jsons

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_gt_doa_wer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== GT-DOA PIPELINE WER/AUDIO SUMMARY =====")
    print(f"Evaluated items: {summary['evaluated_items']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Mean sample WER: {summary['mean_sample_wer']:.4f}")
    print("\nPer-speaker DOA/WER:")
    for spk, spk_summary in by_speaker.items():
        print(
            f"{spk}: "
            f"n={spk_summary['items']}, "
            f"corpus WER={spk_summary['corpus_wer']:.4f}, "
            f"mean WER={spk_summary['mean_sample_wer']:.4f}, "
            f"<=10deg={spk_summary['doa_within_10_deg_percent']:.2f}%, "
            f"<=20deg={spk_summary['doa_within_20_deg_percent']:.2f}%, "
            f"<=30deg={spk_summary['doa_within_30_deg_percent']:.2f}%, "
            f"mean DOA err={spk_summary['mean_doa_error_deg']:.2f}deg"
        )
        print(
            f"  audio: SI-SDRi={spk_summary['mean_sisdri']:.2f}, "
            f"SDRi={spk_summary['mean_sdri']:.2f}, "
            f"PESQ={spk_summary['mean_wb_pesq']:.2f}, "
            f"PESQi={spk_summary['mean_wb_pesqi']:.2f}"
        )
    print(
        "Mean timing per target: "
        f"DSENet={summary['mean_dsenet_sec']:.3f}s, "
        f"metrics={summary['mean_metrics_sec']:.3f}s, "
        f"Whisper={summary['mean_whisper_sec']:.3f}s, "
        f"total={summary['mean_total_target_sec']:.3f}s"
    )
    print(f"Saved WER details: {details_csv}")
    if "spk1" in speaker_detail_csvs:
        print(f"Saved spk1 details: {speaker_detail_csvs['spk1']}")
        print(f"Saved spk1 summary: {speaker_summary_jsons['spk1']}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
