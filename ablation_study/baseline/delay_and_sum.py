"""
Offline IPDNet/GT-DOA -> delay-and-sum beamforming -> Whisper WER benchmark.

For each saved multichannel scene, this script can evaluate two classical
beamforming variants:

1. DSBeamformer-GT: steer once to each ground-truth speaker DOA parsed from the
   target text/clean filenames.
2. DSBeamformer-IPDNet: run IPDNet once, match its three predicted DOAs to the
   three GT speaker DOAs, then steer once to each matched predicted DOA.

No neural enhancement model or output-selection step is used. Each speaker/text
target is evaluated one by one with Whisper and WER.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from scipy.signal import istft, resample_poly, stft
from sklearn.cluster import KMeans
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL" / "IPDNET"
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"

sys.path.insert(0, str(SSL_ROOT))
import FixedAarryIPDnet as ssl_model  # noqa: E402
import Module as ssl_module  # noqa: E402
from utils_ import forgetting_norm  # noqa: E402


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


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_seconds(device: str, fn):
    cuda_sync(device)
    start = time.perf_counter()
    value = fn()
    cuda_sync(device)
    return value, time.perf_counter() - start


def circular_mean_deg_360(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    angles_deg = np.asarray(angles_deg) % 360.0
    angles_rad = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(angles_rad)
    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def circular_angle_error_deg(pred_deg: float, gt_deg: float) -> float:
    return float(abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0))


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


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: Path
    clean_path: Optional[Path]


def load_target_references(text_dir: Path, clean_dir: Path) -> Dict[int, List[TargetReference]]:
    clean_by_key: Dict[Tuple[int, int, int], Path] = {}
    for clean_path in sorted(clean_dir.glob("clean_fileid_*_doa*_spk*.wav")):
        fileid = parse_fileid(clean_path)
        speaker_id = parse_speaker_id(clean_path)
        doa = parse_doa(clean_path)
        clean_by_key[(fileid, speaker_id, doa)] = clean_path

    references: Dict[int, List[TargetReference]] = {}
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk*.txt")):
        fileid = parse_fileid(text_path)
        speaker_id = parse_speaker_id(text_path)
        doa = parse_doa(text_path)
        references.setdefault(fileid, []).append(
            TargetReference(
                fileid=fileid,
                speaker_id=speaker_id,
                gt_doa=doa,
                text_path=text_path,
                clean_path=clean_by_key.get((fileid, speaker_id, doa)),
            )
        )

    for fileid in references:
        references[fileid].sort(key=lambda ref: (ref.speaker_id, ref.gt_doa, ref.text_path.name))
    return references


def postprocess_doa_from_tensors(
    doa_est: torch.Tensor,
    vad_est: torch.Tensor,
    num_sources: int,
    vad_th: float,
) -> List[int]:
    """
    Cluster active SSL DOA points into exactly num_sources DOAs.

    This keeps a fixed three-DOA output whenever enough active SSL points exist
    to form three clusters.
    """
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

    labels = KMeans(n_clusters=num_sources, random_state=0, n_init=10).fit_predict(
        xy,
        sample_weight=valid_weights_np,
    )

    final_doas: List[int] = []
    for source_id in range(num_sources):
        cluster_angles = valid_angles_np[labels == source_id]
        cluster_weights = valid_weights_np[labels == source_id]
        final_doas.append(int(round(circular_mean_deg_360(cluster_angles, cluster_weights))) % 360)

    return sorted(final_doas)


def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


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


def delay_and_sum_beamform(
    wav_tc: np.ndarray,
    sample_rate: int,
    doa_deg: float,
    n_fft: int,
    hop_length: int,
    sound_speed: float,
    mic_positions: np.ndarray = RESPEAKER4_MIC_POS,
) -> np.ndarray:
    if wav_tc.ndim != 2:
        raise ValueError(f"Expected audio shape [samples, channels], got {wav_tc.shape}")
    if wav_tc.shape[1] < 1:
        raise ValueError("Delay-and-sum beamforming requires at least one channel.")
    if hop_length <= 0 or hop_length >= n_fft:
        raise ValueError("--beam_hop must be > 0 and smaller than --beam_n_fft.")

    num_mics = min(wav_tc.shape[1], mic_positions.shape[0])
    if num_mics == 1:
        return wav_tc[:, 0].astype(np.float32)

    theta = np.deg2rad(float(doa_deg) % 360.0)
    unit_vec = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    relative_delays = mic_positions[:num_mics].astype(np.float64) @ unit_vec / float(sound_speed)

    spectra = []
    freqs = None
    for ch in range(num_mics):
        freqs_ch, _, spectrum = stft(
            wav_tc[:, ch],
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            boundary="zeros",
            padded=True,
        )
        if freqs is None:
            freqs = freqs_ch
        spectra.append(spectrum)

    x_fct = np.stack(spectra, axis=1)
    phase = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * relative_delays[None, :])
    y_ft = np.sum(x_fct * phase[:, :, None], axis=1) / float(num_mics)

    _, enhanced = istft(
        y_ft,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
    )
    enhanced = np.nan_to_num(enhanced, copy=False)
    if enhanced.shape[0] < wav_tc.shape[0]:
        enhanced = np.pad(enhanced, (0, wav_tc.shape[0] - enhanced.shape[0]))
    return enhanced[: wav_tc.shape[0]].astype(np.float32)


def match_predicted_doas_to_targets(
    references: Sequence[TargetReference],
    pred_doas: Sequence[int],
) -> Dict[int, Tuple[int, int, float]]:
    if len(pred_doas) < len(references):
        raise ValueError(
            f"Need at least {len(references)} predicted DOAs to match targets, got {len(pred_doas)}."
        )

    best_perm: Optional[Tuple[int, ...]] = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(len(pred_doas)), len(references)):
        cost = sum(
            circular_angle_error_deg(pred_doas[pred_idx], ref.gt_doa)
            for ref, pred_idx in zip(references, perm)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        return {}

    matched: Dict[int, Tuple[int, int, float]] = {}
    for ref, pred_idx in zip(references, best_perm):
        pred_doa = int(pred_doas[pred_idx])
        matched[ref.speaker_id] = (
            pred_idx,
            pred_doa,
            circular_angle_error_deg(pred_doa, ref.gt_doa),
        )
    return matched


@dataclass
class AsrTiming:
    fileid: int
    mic_file: str
    method: str
    doa_source: str
    speaker_id: int
    duration_sec: float
    gt_doa: int
    steering_doa: int
    predicted_doa_count: int
    predicted_doas: str
    matched_predicted_index: Optional[int]
    steering_doa_error_deg: Optional[float]
    beamformed_file: str
    gt_text_file: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    reference: str
    hypothesis: str
    ipdnet_sec: float
    beamforming_sec: float
    whisper_sec: float
    total_sec: float
    ipdnet_rtf: float
    beamforming_rtf: float
    whisper_rtf: float
    total_rtf: float
    under_realtime: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GT/IPDNet DOA -> delay-and-sum -> Whisper WER benchmark.")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "last-v1.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--vad_th", type=float, default=0.7)
    parser.add_argument(
        "--doa_source",
        choices=("gt", "ipdnet", "both"),
        default="both",
        help="Evaluate GT filename DOAs, IPDNet-predicted DOAs, or both beamforming versions.",
    )
    parser.add_argument("--beam_n_fft", type=int, default=512)
    parser.add_argument("--beam_hop", type=int, default=256)
    parser.add_argument("--sound_speed", type=float, default=343.0)
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save each beamformed wav.")
    return parser.parse_args()


def append_result_for_target(
    *,
    args: argparse.Namespace,
    whisper_model,
    enhanced_dir: Path,
    fileid: int,
    mic_file: str,
    duration_sec: float,
    sr: int,
    ref: TargetReference,
    method: str,
    doa_source: str,
    steering_doa: int,
    pred_doas: Sequence[int],
    matched_pred_idx: Optional[int],
    steering_error: Optional[float],
    enhanced_for_asr: np.ndarray,
    ipd_sec: float,
    beam_sec: float,
) -> AsrTiming:
    beamformed_name = (
        f"beamformed_fileid_{fileid}_spk{ref.speaker_id}_"
        f"{doa_source}_doa{steering_doa}_gt{ref.gt_doa}.wav"
    )

    if args.save_enhanced:
        sf.write(str(enhanced_dir / beamformed_name), enhanced_for_asr, sr)

    def run_asr():
        return whisper_model.transcribe(
            enhanced_for_asr,
            language=args.language,
            fp16=args.whisper_device.startswith("cuda"),
        )

    asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
    hyp_text = asr_out.get("text", "").strip()
    ref_text = ref.text_path.read_text(encoding="utf-8").strip()
    sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)

    total_sec = ipd_sec + beam_sec + whisper_sec
    return AsrTiming(
        fileid=fileid,
        mic_file=mic_file,
        method=method,
        doa_source=doa_source,
        speaker_id=ref.speaker_id,
        duration_sec=duration_sec,
        gt_doa=ref.gt_doa,
        steering_doa=steering_doa,
        predicted_doa_count=len(pred_doas),
        predicted_doas=",".join(str(doa) for doa in pred_doas),
        matched_predicted_index=matched_pred_idx,
        steering_doa_error_deg=steering_error,
        beamformed_file=beamformed_name,
        gt_text_file=ref.text_path.name,
        wer=sample_wer,
        edit_distance=dist,
        ref_words=ref_word_count,
        reference=ref_text,
        hypothesis=hyp_text,
        ipdnet_sec=ipd_sec,
        beamforming_sec=beam_sec,
        whisper_sec=whisper_sec,
        total_sec=total_sec,
        ipdnet_rtf=ipd_sec / duration_sec,
        beamforming_rtf=beam_sec / duration_sec,
        whisper_rtf=whisper_sec / duration_sec,
        total_rtf=total_sec / duration_sec,
        under_realtime=int(total_sec < duration_sec),
    )


def summarize_method(rows: Sequence[AsrTiming]) -> Dict[str, float | int]:
    evaluated = [row for row in rows if row.wer is not None]
    total_edits = sum(int(row.edit_distance or 0) for row in evaluated)
    total_ref_words = sum(int(row.ref_words or 0) for row in evaluated)
    wers = [float(row.wer) for row in evaluated]
    doa_errors = [
        float(row.steering_doa_error_deg)
        for row in evaluated
        if row.steering_doa_error_deg is not None
    ]
    return {
        "evaluated_utterances": len(evaluated),
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
        "mean_sample_wer": float(np.mean(wers)) if wers else 0.0,
        "mean_steering_doa_error_deg": float(np.mean(doa_errors)) if doa_errors else 0.0,
        "median_steering_doa_error_deg": float(np.median(doa_errors)) if doa_errors else 0.0,
        "mean_ipdnet_sec": float(np.mean([row.ipdnet_sec for row in evaluated])) if evaluated else 0.0,
        "mean_beamforming_sec": float(np.mean([row.beamforming_sec for row in evaluated])) if evaluated else 0.0,
        "mean_whisper_sec": float(np.mean([row.whisper_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_sec": float(np.mean([row.total_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_rtf": float(np.mean([row.total_rtf for row in evaluated])) if evaluated else 0.0,
        "under_realtime_count": int(sum(row.under_realtime for row in evaluated)),
        "under_realtime_rate": float(np.mean([row.under_realtime for row in evaluated])) if evaluated else 0.0,
    }


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")
    if args.beam_hop <= 0 or args.beam_hop >= args.beam_n_fft:
        raise ValueError("--beam_hop must be > 0 and smaller than --beam_n_fft.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_delay_sum_enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    needs_ipdnet = args.doa_source in ("ipdnet", "both")

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"DOA source: {args.doa_source}")
    print(f"Beamformer: delay-and-sum, n_fft={args.beam_n_fft}, hop={args.beam_hop}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Target text folder: {args.text_dir}")

    ipd_model: Optional[IPDNetInference] = None
    if needs_ipdnet:
        print("Loading IPDNET once...")
        ipd_model = load_ipdnet(args.ipd_ckpt, args.device)

    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    references_by_fileid = load_target_references(args.text_dir, args.clean_dir)
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print(f"Target text references: {sum(len(v) for v in references_by_fileid.values())}")

    scene_results: List[AsrTiming] = []
    skipped_no_gt_targets = 0
    skipped_no_ipdnet_doa = 0
    truncated_target_refs = 0

    for fileid, target_paths in tqdm(grouped.items(), desc="DelaySum-ASR", unit="scene"):
        refs = references_by_fileid.get(fileid, [])
        if not refs:
            skipped_no_gt_targets += 1
            print(f"fileid={fileid}: no text_fileid_*_doa*_spk*.txt references, skipped.")
            continue
        if len(refs) > args.num_sources:
            refs = refs[: args.num_sources]
            truncated_target_refs += 1
        if len(refs) < args.num_sources:
            print(
                f"fileid={fileid}: found {len(refs)} target references, "
                f"expected {args.num_sources}; evaluating available targets."
            )

        mic_path = choose_representative_mic(target_paths)
        wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
        duration_sec = wav_tc.shape[0] / float(sr)
        pred_doas: List[int] = []
        ipd_sec = 0.0

        if needs_ipdnet:
            assert ipd_model is not None
            mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)

            def run_ssl():
                with torch.inference_mode():
                    return ipd_model(mic_batch)

            ssl_out, ipd_sec = elapsed_seconds(args.device, run_ssl)
            pred_doas = postprocess_doa_from_tensors(
                ssl_out["doa_est"],
                ssl_out["vad_est"],
                num_sources=args.num_sources,
                vad_th=args.vad_th,
            )

        if args.doa_source in ("gt", "both"):
            for ref in refs:
                enhanced, beam_sec = elapsed_seconds(
                    "cpu",
                    lambda ref=ref: delay_and_sum_beamform(
                        wav_tc,
                        sr,
                        doa_deg=ref.gt_doa,
                        n_fft=args.beam_n_fft,
                        hop_length=args.beam_hop,
                        sound_speed=args.sound_speed,
                    ),
                )
                scene_results.append(
                    append_result_for_target(
                        args=args,
                        whisper_model=whisper_model,
                        enhanced_dir=enhanced_dir,
                        fileid=fileid,
                        mic_file=mic_path.name,
                        duration_sec=duration_sec,
                        sr=sr,
                        ref=ref,
                        method="DSBeamformer-GT",
                        doa_source="gt",
                        steering_doa=ref.gt_doa,
                        pred_doas=pred_doas,
                        matched_pred_idx=None,
                        steering_error=0.0,
                        enhanced_for_asr=enhanced,
                        ipd_sec=0.0,
                        beam_sec=beam_sec,
                    )
                )

        if args.doa_source in ("ipdnet", "both"):
            if len(pred_doas) != args.num_sources:
                skipped_no_ipdnet_doa += 1
                print(
                    f"fileid={fileid}: expected {args.num_sources} IPDNet DOAs, "
                    f"got {len(pred_doas)}, skipped IPDNet-DOA beamforming."
                )
                continue

            matched = match_predicted_doas_to_targets(refs, pred_doas)
            for ref in refs:
                matched_pred_idx, steering_doa, steering_error = matched[ref.speaker_id]
                enhanced, beam_sec = elapsed_seconds(
                    "cpu",
                    lambda steering_doa=steering_doa: delay_and_sum_beamform(
                        wav_tc,
                        sr,
                        doa_deg=steering_doa,
                        n_fft=args.beam_n_fft,
                        hop_length=args.beam_hop,
                        sound_speed=args.sound_speed,
                    ),
                )
                scene_results.append(
                    append_result_for_target(
                        args=args,
                        whisper_model=whisper_model,
                        enhanced_dir=enhanced_dir,
                        fileid=fileid,
                        mic_file=mic_path.name,
                        duration_sec=duration_sec,
                        sr=sr,
                        ref=ref,
                        method="DSBeamformer-IPDNet",
                        doa_source="ipdnet",
                        steering_doa=steering_doa,
                        pred_doas=pred_doas,
                        matched_pred_idx=matched_pred_idx,
                        steering_error=steering_error,
                        enhanced_for_asr=enhanced,
                        ipd_sec=ipd_sec,
                        beam_sec=beam_sec,
                    )
                )

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_dsbeamformer_wer_details_1asr.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(AsrTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    by_method = {
        method: summarize_method([row for row in scene_results if row.method == method])
        for method in sorted({row.method for row in scene_results})
    }
    all_summary = summarize_method(scene_results)

    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
        "text_dir": str(args.text_dir),
        "ipd_ckpt": str(args.ipd_ckpt) if needs_ipdnet else "",
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "enhancement": "delay_and_sum_beamforming",
        "doa_source": args.doa_source,
        "beam_n_fft": args.beam_n_fft,
        "beam_hop": args.beam_hop,
        "sound_speed": args.sound_speed,
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "target_text_references": sum(len(v) for v in references_by_fileid.values()),
        "evaluated_utterances": len(scene_results),
        "skipped_no_gt_targets": skipped_no_gt_targets,
        "skipped_no_ipdnet_doa": skipped_no_ipdnet_doa,
        "truncated_target_ref_groups": truncated_target_refs,
        "overall": all_summary,
        "by_method": by_method,
    }

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_dsbeamformer_wer_summary_1asr.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== DELAY-AND-SUM ASR SUMMARY =====")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Skipped scenes without target text refs: {summary['skipped_no_gt_targets']}")
    print(f"Skipped IPDNet-DOA scene variants: {summary['skipped_no_ipdnet_doa']}")
    for method, stats in by_method.items():
        print(
            f"{method}: utterances={stats['evaluated_utterances']}, "
            f"corpus WER={stats['corpus_wer']:.4f}, "
            f"mean sample WER={stats['mean_sample_wer']:.4f}, "
            f"mean DOA error={stats['mean_steering_doa_error_deg']:.2f} deg, "
            f"mean total RTF={stats['mean_total_rtf']:.3f}"
        )
    print(f"Saved details: {details_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
