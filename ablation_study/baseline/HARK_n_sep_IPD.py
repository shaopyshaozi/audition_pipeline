"""
Official HARK + IPDNet-DOA baseline wrapper and evaluator.

This script uses IPDNet to predict source DOAs, injects those DOAs into an
official HARK ConstantLocalization + GHDSS separation network, and evaluates
the separated output whose predicted DOA is closest to spk1's ground-truth DOA:

    4-channel wav
        -> IPDNet + KMeans DOA clustering
        -> official HARK network:
           ConstantLocalization with IPDNet-predicted DOAs -> GHDSS
        -> separated wavs labeled by predicted DOA
        -> Whisper small
        -> WER / SDRi / SI-SDRi

Usage is intentionally two-stage:

1. run_official:
   patch and call an official HARK .n network once per test scene.
2. eval_official:
   evaluate the separated wavs produced by official HARK with the same Whisper
   and metric code used by the other baselines.

The .n file is used as a template. The script patches its input wav path,
ConstantLocalization azimuth/elevation vectors, TF path, and output basename.
Evaluation compares spk1 only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import string
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL" / "IPDNET"

sys.path.insert(0, str(SSL_ROOT))
import FixedAarryIPDnet as ssl_model  # noqa: E402
import Module as ssl_module  # noqa: E402
from utils_ import forgetting_norm  # noqa: E402


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


def circular_angle_error_deg(pred_deg: float, gt_deg: float) -> float:
    return float(abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0))


def circular_mean_deg_360(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    angles_deg = np.asarray(angles_deg) % 360.0
    angles_rad = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(angles_rad)
    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


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


def nearest_predicted_doa_index(pred_doas: Sequence[int], target_doa: int) -> Optional[int]:
    if not pred_doas:
        return None
    return min(range(len(pred_doas)), key=lambda idx: circular_angle_error_deg(pred_doas[idx], target_doa))


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


def si_sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred = pred - np.mean(pred)
    target = target - np.mean(target)
    scale = np.dot(pred, target) / (np.dot(target, target) + eps)
    projection = scale * target
    noise = pred - projection
    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)))


def sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    error = target - pred
    return float(10.0 * np.log10((np.sum(target**2) + eps) / (np.sum(error**2) + eps)))


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: Path
    clean_path: Optional[Path]


@dataclass
class HarkRunRecord:
    fileid: int
    mic_file: str
    predicted_doas: str
    output_dir: str
    output_prefix: str
    command: str
    returncode: int
    elapsed_sec: float
    stdout_log: str
    stderr_log: str
    separated_wav_count: int


@dataclass
class EvalRecord:
    fileid: int
    mic_file: str
    method: str
    speaker_id: int
    duration_sec: float
    gt_doa: int
    selected_wav: str
    selection_strategy: str
    selected_source_index: Optional[int]
    predicted_doa: Optional[float]
    doa_error_deg: Optional[float]
    separated_wav_count: int
    gt_text_file: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    input_sdr: Optional[float]
    sdr: Optional[float]
    sdr_i: Optional[float]
    input_si_sdr: Optional[float]
    si_sdr: Optional[float]
    si_sdr_i: Optional[float]
    reference: str
    hypothesis: str
    whisper_sec: float
    total_sec: float
    whisper_rtf: float
    total_rtf: float
    under_realtime: int


def load_scene_references(
    text_dir: Path,
    clean_dir: Path,
    target_speaker_id: int,
) -> Tuple[Dict[int, List[TargetReference]], Dict[int, List[TargetReference]]]:
    clean_by_key: Dict[Tuple[int, int, int], Path] = {}
    for clean_path in sorted(clean_dir.glob("clean_fileid_*_doa*_spk*.wav")):
        fileid = parse_fileid(clean_path)
        speaker_id = parse_speaker_id(clean_path)
        doa = parse_doa(clean_path)
        clean_by_key[(fileid, speaker_id, doa)] = clean_path

    scene_refs: Dict[int, List[TargetReference]] = {}
    target_refs: Dict[int, List[TargetReference]] = {}
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk*.txt")):
        fileid = parse_fileid(text_path)
        speaker_id = parse_speaker_id(text_path)
        doa = parse_doa(text_path)
        ref = TargetReference(
            fileid=fileid,
            speaker_id=speaker_id,
            gt_doa=doa,
            text_path=text_path,
            clean_path=clean_by_key.get((fileid, speaker_id, doa)),
        )
        scene_refs.setdefault(fileid, []).append(ref)
        if target_speaker_id == 0 or speaker_id == target_speaker_id:
            target_refs.setdefault(fileid, []).append(ref)

    for refs in scene_refs.values():
        refs.sort(key=lambda ref: (ref.speaker_id, ref.gt_doa, ref.text_path.name))
    for refs in target_refs.values():
        refs.sort(key=lambda ref: (ref.speaker_id, ref.gt_doa, ref.text_path.name))
    return scene_refs, target_refs


def compute_audio_quality_metrics(
    *,
    enhanced: np.ndarray,
    clean_path: Optional[Path],
    noisy_ref: np.ndarray,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    empty = {
        "input_sdr": None,
        "sdr": None,
        "sdr_i": None,
        "input_si_sdr": None,
        "si_sdr": None,
        "si_sdr_i": None,
    }
    if clean_path is None:
        return empty

    clean, clean_sr = load_mono_audio(clean_path, target_sr=sample_rate)
    if clean_sr != sample_rate:
        return empty

    min_len = min(len(enhanced), len(clean), len(noisy_ref))
    if min_len <= 0:
        return empty

    enhanced = enhanced[:min_len]
    clean = clean[:min_len]
    noisy_ref = noisy_ref[:min_len]
    input_sdr = sdr_db(noisy_ref, clean)
    output_sdr = sdr_db(enhanced, clean)
    input_si_sdr = si_sdr_db(noisy_ref, clean)
    output_si_sdr = si_sdr_db(enhanced, clean)
    return {
        "input_sdr": input_sdr,
        "sdr": output_sdr,
        "sdr_i": output_sdr - input_sdr,
        "input_si_sdr": input_si_sdr,
        "si_sdr": output_si_sdr,
        "si_sdr_i": output_si_sdr - input_si_sdr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call official HARK and evaluate its separated wavs.")
    parser.add_argument(
        "--mode",
        choices=("run_official", "eval_official", "both"),
        default="both",
        help="Run official HARK, evaluate existing official HARK outputs, or do both.",
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--official_output_dir", type=Path, default=None)
    parser.add_argument(
        "--hark_network",
        type=Path,
        default=OFFLINE_ROOT / "HARK" / "sep.n",
        help="Official HARK .n network file. This IPD-DOA script expects a ConstantLocalization node.",
    )
    parser.add_argument("--hark_runner", type=str, default="batchflow")
    parser.add_argument(
        "--tf_zip",
        type=Path,
        default=OFFLINE_ROOT / "HARK" / "respeaker4_tf_5deg.zip",
        help="Transfer-function zip used to patch A_MATRIX and TF_CONJ_FILENAME in the runtime .n file.",
    )
    parser.add_argument(
        "--hark_command_template",
        type=str,
        default="{runner} {network} {input_wav} {output_prefix}",
        help=(
            "Command template for official HARK. Available fields: {runner}, {network}, "
            "{input_wav}, {output_prefix}, {output_dir}, {fileid}, {mic_stem}."
        ),
    )
    parser.add_argument(
        "--official_output_glob",
        type=str,
        default="*.wav",
        help="Glob, relative to each scene output folder, used to find HARK separated wavs.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="In run_official mode, skip scenes that already have wavs matching --official_output_glob.",
    )
    parser.add_argument(
        "--source_selection",
        choices=("nearest_spk1_gt_doa",),
        default="nearest_spk1_gt_doa",
        help=(
            "Choose the enhanced stream whose injected IPDNet-predicted DOA is closest "
            "to the spk1 ground-truth DOA."
        ),
    )
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "last-v1.ckpt")
    parser.add_argument("--ipd_device", type=str, default="cuda")
    parser.add_argument("--vad_th", type=float, default=0.7)
    parser.add_argument("--min_points_per_source", type=int, default=3)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument(
        "--target_speaker_id",
        type=int,
        default=1,
        choices=(1,),
        help="Speaker id to evaluate. This IPD separation script compares spk1 only.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    return parser.parse_args()


def scene_output_dir(args: argparse.Namespace, fileid: int) -> Path:
    root = args.official_output_dir or (args.out_dir / "official_hark_outputs")
    return root / f"fileid_{fileid}"


def scene_output_prefix(args: argparse.Namespace, fileid: int, mic_path: Path) -> Path:
    return scene_output_dir(args, fileid) / f"hark_raw_fileid_{fileid}"


def find_scene_wavs(args: argparse.Namespace, fileid: int) -> List[Path]:
    out_dir = scene_output_dir(args, fileid)
    if not out_dir.exists():
        return []
    return sorted(path for path in out_dir.glob(args.official_output_glob) if path.is_file())


def clear_scene_wavs(args: argparse.Namespace, fileid: int) -> None:
    for wav_path in find_scene_wavs(args, fileid):
        wav_path.unlink()


def ipd_candidate_output_path(args: argparse.Namespace, fileid: int, pred_doa: int, source_index: int) -> Path:
    return scene_output_dir(args, fileid) / (
        f"enhanced_fileid_{fileid}_preddoa{pred_doa}_src{source_index + 1}.wav"
    )


def sort_hark_outputs(wavs: Sequence[Path]) -> List[Path]:
    def key(path: Path) -> Tuple[int, str]:
        match = re.search(r"_(\d+)\.wav$", path.name)
        return (int(match.group(1)) if match else 10_000, path.name)

    return sorted(wavs, key=key)


def find_raw_hark_wavs(args: argparse.Namespace, fileid: int) -> List[Path]:
    wavs = find_scene_wavs(args, fileid)
    return sort_hark_outputs(
        [
            path
            for path in wavs
            if not path.name.startswith("enhanced_fileid_")
            and not path.name.startswith("selected_fileid_")
        ]
    )


def relabel_hark_outputs_in_place(
    args: argparse.Namespace,
    fileid: int,
    pred_doas: Sequence[int],
) -> None:
    raw_wavs = find_raw_hark_wavs(args, fileid)
    for source_index, pred_doa in enumerate(pred_doas):
        if source_index < 0 or source_index >= len(raw_wavs):
            continue
        source_path = raw_wavs[source_index]
        target_path = ipd_candidate_output_path(args, fileid, int(pred_doa), source_index)
        if source_path.resolve() == target_path.resolve():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source_path.replace(target_path)


def candidate_wavs_for_pred_doas(
    args: argparse.Namespace,
    fileid: int,
    pred_doas: Sequence[int],
) -> List[Tuple[Path, int, int]]:
    candidates: List[Tuple[Path, int, int]] = []
    for source_index, pred_doa in enumerate(pred_doas):
        path = ipd_candidate_output_path(args, fileid, int(pred_doa), source_index)
        if path.is_file():
            candidates.append((path, source_index, int(pred_doa)))
    return candidates


def parse_predicted_doas_from_candidate_wavs(args: argparse.Namespace, fileid: int) -> List[int]:
    values: List[Tuple[int, int]] = []
    for wav_path in find_scene_wavs(args, fileid):
        match = re.search(r"preddoa(\d+)_src(\d+)\.wav$", wav_path.name)
        if not match:
            continue
        values.append((int(match.group(2)) - 1, int(match.group(1))))
    return [pred for _, pred in sorted(values)]


def vector_float(values: Sequence[float | int]) -> str:
    body = " ".join(str(int(value)) if float(value).is_integer() else str(float(value)) for value in values)
    return f"<Vector<float> {body}>"


def predict_ipd_doas_for_scene(
    args: argparse.Namespace,
    ipd_model: IPDNetInference,
    mic_path: Path,
) -> List[int]:
    wav_tc, _ = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
    mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)

    with torch.no_grad():
        ssl_out = ipd_model(mic_batch)
    return postprocess_doa_from_tensors(
        ssl_out["doa_est"],
        ssl_out["vad_est"],
        num_sources=args.num_sources,
        vad_th=args.vad_th,
        min_points_per_source=args.min_points_per_source,
    )


def load_predicted_doa_manifest(args: argparse.Namespace) -> Dict[int, List[int]]:
    manifest_csv = args.out_dir / "official_hark_run_manifest.csv"
    if not manifest_csv.is_file():
        return {}
    predicted: Dict[int, List[int]] = {}
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values = row.get("predicted_doas") or ""
            if not values:
                continue
            predicted[int(row["fileid"])] = [int(float(value)) for value in values.split(",") if value != ""]
    return predicted


def patch_hark_network_for_scene(
    args: argparse.Namespace,
    mic_path: Path,
    fileid: int,
    pred_doas: Sequence[int],
) -> Path:
    if args.hark_network is None:
        raise ValueError("--hark_network is required for --mode run_official or --mode both.")
    if not pred_doas:
        raise ValueError(f"No predicted DOAs available for fileid={fileid}.")

    out_dir = scene_output_dir(args, fileid)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_network = out_dir / f"runtime_fileid_{fileid}.n"
    output_basename = str(scene_output_prefix(args, fileid, mic_path)) + "_"

    text = args.hark_network.read_text(encoding="utf-8")
    shebang = ""
    if text.startswith("#!"):
        first_newline = text.find("\n")
        shebang = text[: first_newline + 1]
        text = text[first_newline + 1 :]

    root = ET.fromstring(text)
    tf_zip = args.tf_zip
    if tf_zip is not None and not tf_zip.is_file():
        raise FileNotFoundError(f"Transfer-function zip not found: {tf_zip}")

    for node in root.iter("Node"):
        node_type = node.attrib.get("type")
        for parameter in node.findall("Parameter"):
            name = parameter.attrib.get("name")
            if node_type == "Constant" and name == "VALUE":
                parameter.set("value", str(mic_path))
            elif node_type == "ConstantLocalization" and name == "ANGLES":
                parameter.set("value", vector_float(pred_doas))
            elif node_type == "ConstantLocalization" and name == "ELEVATIONS":
                parameter.set("value", vector_float([0] * len(pred_doas)))
            elif node_type == "SaveWavePCM" and name == "BASENAME":
                parameter.set("value", output_basename)
            elif name in {"A_MATRIX", "TF_CONJ_FILENAME"} and tf_zip is not None:
                parameter.set("value", str(tf_zip))

    xml_text = ET.tostring(root, encoding="unicode")
    runtime_network.write_text(shebang + '<?xml version="1.0"?>\n' + xml_text + "\n", encoding="utf-8")
    return runtime_network


def render_hark_command(
    args: argparse.Namespace,
    mic_path: Path,
    fileid: int,
    pred_doas: Sequence[int],
) -> List[str]:
    if args.hark_network is None:
        raise ValueError("--hark_network is required for --mode run_official or --mode both.")

    out_dir = scene_output_dir(args, fileid)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = scene_output_prefix(args, fileid, mic_path)
    runtime_network = patch_hark_network_for_scene(args, mic_path, fileid, pred_doas)
    values = {
        "runner": str(args.hark_runner),
        "network": str(runtime_network),
        "input_wav": str(mic_path),
        "output_prefix": str(output_prefix),
        "output_dir": str(out_dir),
        "fileid": str(fileid),
        "mic_stem": mic_path.stem,
    }
    command = args.hark_command_template.format(**values)
    return shlex.split(command)


def run_official_hark(args: argparse.Namespace) -> List[HarkRunRecord]:
    if args.hark_network is None:
        raise FileNotFoundError("--hark_network is required in run_official/both mode.")
    if not args.hark_network.is_file():
        raise FileNotFoundError(
            f"HARK network file not found: {args.hark_network}\n"
            "Replace the example path with your real official HARK .n file."
        )
    runner_path = shutil.which(args.hark_runner)
    if runner_path is None and not Path(args.hark_runner).is_file():
        raise FileNotFoundError(
            f"HARK runner not found on PATH: {args.hark_runner}\n"
            "Install official HARK in this Ubuntu/WSL environment, then verify with:\n"
            f"  which {args.hark_runner}\n"
            "If your HARK runner has a different name/path, pass --hark_runner /path/to/runner "
            "or adjust --hark_command_template."
        )

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    records: List[HarkRunRecord] = []
    predicted_manifest = load_predicted_doa_manifest(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "official_hark_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.ipd_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("IPDNet was requested on CUDA, but torch.cuda.is_available() is False.")
    print(f"Loading IPDNet once: {args.ipd_ckpt} on {args.ipd_device}")
    ipd_model = load_ipdnet(args.ipd_ckpt, args.ipd_device)

    for fileid, target_paths in tqdm(grouped.items(), desc="Official-HARK", unit="scene"):
        mic_path = choose_representative_mic(target_paths)
        pred_doas = predicted_manifest.get(fileid)
        if pred_doas is None:
            pred_doas = predict_ipd_doas_for_scene(args, ipd_model, mic_path)
        if not pred_doas:
            print(f"fileid={fileid}: no usable IPDNet DOA, skipped HARK run.")
            continue
        out_dir = scene_output_dir(args, fileid)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = find_scene_wavs(args, fileid)
        if args.skip_existing and existing and len(candidate_wavs_for_pred_doas(args, fileid, pred_doas)) >= len(pred_doas):
            records.append(
                HarkRunRecord(
                    fileid=fileid,
                    mic_file=str(mic_path),
                    predicted_doas=",".join(str(doa) for doa in pred_doas),
                    output_dir=str(out_dir),
                    output_prefix=str(scene_output_prefix(args, fileid, mic_path)),
                    command="SKIPPED_EXISTING",
                    returncode=0,
                    elapsed_sec=0.0,
                    stdout_log="",
                    stderr_log="",
                    separated_wav_count=len(existing),
                )
            )
            continue

        clear_scene_wavs(args, fileid)
        command = render_hark_command(args, mic_path, fileid, pred_doas)
        stdout_log = logs_dir / f"hark_fileid_{fileid}.stdout.log"
        stderr_log = logs_dir / f"hark_fileid_{fileid}.stderr.log"
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode == 0:
            relabel_hark_outputs_in_place(args, fileid, pred_doas)
        wavs = find_scene_wavs(args, fileid)
        records.append(
            HarkRunRecord(
                fileid=fileid,
                mic_file=str(mic_path),
                predicted_doas=",".join(str(doa) for doa in pred_doas),
                output_dir=str(out_dir),
                output_prefix=str(scene_output_prefix(args, fileid, mic_path)),
                command=" ".join(shlex.quote(part) for part in command),
                returncode=int(proc.returncode),
                elapsed_sec=elapsed,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                separated_wav_count=len(wavs),
            )
        )

    manifest_csv = args.out_dir / "official_hark_run_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HarkRunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))
    return records


def evaluate_official_outputs(args: argparse.Namespace) -> List[EvalRecord]:
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")

    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    _, target_refs_by_fileid = load_scene_references(
        args.text_dir,
        args.clean_dir,
        target_speaker_id=args.target_speaker_id,
    )
    predicted_manifest = load_predicted_doa_manifest(args)
    records: List[EvalRecord] = []
    skipped = {
        "no_target_refs": 0,
        "no_hark_wavs": 0,
        "selection_failed": 0,
    }

    for fileid, target_paths in tqdm(grouped.items(), desc="Eval-official-HARK", unit="scene"):
        target_refs = target_refs_by_fileid.get(fileid, [])
        if not target_refs:
            skipped["no_target_refs"] += 1
            continue

        mic_path = choose_representative_mic(target_paths)
        wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
        noisy_ref = wav_tc[:, 0]
        duration_sec = len(noisy_ref) / float(sr)
        separated_wavs = find_scene_wavs(args, fileid)
        if not separated_wavs:
            skipped["no_hark_wavs"] += 1
            continue
        pred_doas = predicted_manifest.get(fileid) or parse_predicted_doas_from_candidate_wavs(args, fileid)
        candidates = candidate_wavs_for_pred_doas(args, fileid, pred_doas)
        if not candidates:
            skipped["no_hark_wavs"] += 1
            continue

        for ref in target_refs:
            ref_text = ref.text_path.read_text(encoding="utf-8").strip()
            try:
                selected_candidate_index = nearest_predicted_doa_index(pred_doas, ref.gt_doa)
                if selected_candidate_index is None:
                    raise ValueError(
                        f"No IPDNet-predicted DOA available for spk{ref.speaker_id}, doa{ref.gt_doa}."
                    )
                selected_pred_doa = int(pred_doas[selected_candidate_index])
                selected_items = [
                    item for item in candidates if item[1] == selected_candidate_index and item[2] == selected_pred_doa
                ]
                if not selected_items:
                    raise ValueError(
                        f"No HARK output wav for predicted DOA {selected_pred_doa} "
                        f"at source index {selected_candidate_index}."
                    )
                selected_wav, selected_index, pred_doa = selected_items[0]
                enhanced, _ = load_mono_audio(selected_wav, target_sr=sr)

                def run_asr():
                    return whisper_model.transcribe(
                        enhanced,
                        language=args.language,
                        fp16=args.whisper_device.startswith("cuda"),
                    )

                asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
                hyp_text = asr_out.get("text", "").strip()
                sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)
                doa_error = circular_angle_error_deg(pred_doa, ref.gt_doa)
            except Exception as exc:
                skipped["selection_failed"] += 1
                print(f"fileid={fileid}, spk{ref.speaker_id}: output selection failed: {exc}")
                continue

            metrics = compute_audio_quality_metrics(
                enhanced=enhanced,
                clean_path=ref.clean_path,
                noisy_ref=noisy_ref,
                sample_rate=sr,
            )
            records.append(
                EvalRecord(
                    fileid=fileid,
                    mic_file=str(mic_path),
                    method="Official-HARK-IPDNetDOA-ConstantLocalization-GHDSS",
                    speaker_id=ref.speaker_id,
                    duration_sec=duration_sec,
                    gt_doa=ref.gt_doa,
                    selected_wav=str(selected_wav),
                    selection_strategy=args.source_selection,
                    selected_source_index=selected_index,
                    predicted_doa=pred_doa,
                    doa_error_deg=doa_error,
                    separated_wav_count=len(separated_wavs),
                    gt_text_file=str(ref.text_path),
                    wer=sample_wer,
                    edit_distance=dist,
                    ref_words=ref_word_count,
                    input_sdr=metrics["input_sdr"],
                    sdr=metrics["sdr"],
                    sdr_i=metrics["sdr_i"],
                    input_si_sdr=metrics["input_si_sdr"],
                    si_sdr=metrics["si_sdr"],
                    si_sdr_i=metrics["si_sdr_i"],
                    reference=ref_text,
                    hypothesis=hyp_text,
                    whisper_sec=whisper_sec,
                    total_sec=whisper_sec,
                    whisper_rtf=whisper_sec / duration_sec,
                    total_rtf=whisper_sec / duration_sec,
                    under_realtime=int(whisper_sec < duration_sec),
                )
            )

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_official_hark_wer_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))

    summary = summarize_eval(records)
    summary.update(
        {
            "mic_dir": str(args.mic_dir),
            "clean_dir": str(args.clean_dir),
            "text_dir": str(args.text_dir),
            "official_output_dir": str(args.official_output_dir or (args.out_dir / "official_hark_outputs")),
            "official_output_glob": args.official_output_glob,
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "target_speaker_id": args.target_speaker_id,
            "source_selection": args.source_selection,
            "official_hark_runtime": True,
            "skipped": skipped,
        }
    )
    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_official_hark_wer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== OFFICIAL HARK ASR SUMMARY =====")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Mean sample WER: {summary['mean_sample_wer']:.4f}")
    print(f"Mean SDRi: {summary['mean_sdri']:.4f}")
    print(f"Mean SI-SDRi: {summary['mean_sisdri']:.4f}")
    print(f"Saved details: {details_csv}")
    print(f"Saved summary: {summary_json}")
    return records


def summarize_eval(rows: Sequence[EvalRecord]) -> Dict[str, float | int]:
    total_edits = sum(int(row.edit_distance or 0) for row in rows)
    total_ref_words = sum(int(row.ref_words or 0) for row in rows)
    wers = [float(row.wer) for row in rows if row.wer is not None]

    def mean_optional(name: str) -> float:
        values = [
            float(value)
            for row in rows
            for value in [getattr(row, name)]
            if value is not None and np.isfinite(float(value))
        ]
        return float(np.mean(values)) if values else 0.0

    return {
        "evaluated_utterances": len(rows),
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
        "mean_sample_wer": float(np.mean(wers)) if wers else 0.0,
        "mean_input_sdr": mean_optional("input_sdr"),
        "mean_sdr": mean_optional("sdr"),
        "mean_sdri": mean_optional("sdr_i"),
        "mean_input_si_sdr": mean_optional("input_si_sdr"),
        "mean_si_sdr": mean_optional("si_sdr"),
        "mean_sisdri": mean_optional("si_sdr_i"),
        "mean_whisper_sec": mean_optional("whisper_sec"),
        "mean_total_rtf": mean_optional("total_rtf"),
        "under_realtime_count": int(sum(row.under_realtime for row in rows)),
        "under_realtime_rate": float(np.mean([row.under_realtime for row in rows])) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("run_official", "both"):
        run_records = run_official_hark(args)
        failed = [row for row in run_records if row.returncode != 0]
        if failed:
            print(f"Official HARK failed on {len(failed)} scenes. Check official_hark_logs.")
            if args.mode == "both":
                raise RuntimeError("Stopping before evaluation because at least one HARK run failed.")

    if args.mode in ("eval_official", "both"):
        evaluate_official_outputs(args)


if __name__ == "__main__":
    main()
