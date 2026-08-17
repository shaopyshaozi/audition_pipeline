"""
Offline HARK-style robot-audition baseline on the 4-mic/3-speaker test set.

Pipeline:

    multichannel wav
        -> LocalizeMUSIC-style DOA estimation
        -> SourceTracker-lite static source selection
        -> GHDSS-style geometric/decorrelation separation
        -> Whisper small ASR
        -> WER / SI-SDRi / SDRi / PESQ summary

This script deliberately does not use GT DOAs or IPDNet DOAs. It is arranged
after the original HARK architecture: MUSIC localization feeds GHDSS
separation, and only the ASR backend is replaced with Whisper for a controlled
comparison with the rest of this repository.

Important: this is a local Python implementation of the HARK architecture, not
a bit-exact official HARK/PyHARK runtime. If official HARK is installed later,
its separated wavs can be evaluated with the same Whisper/metric functions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import find_peaks, istft, resample_poly, stft
from tqdm import tqdm

OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
SCRIPT_STEM = Path(__file__).stem

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


def group_targets_by_fileid(mic_files: Sequence[Path]) -> Dict[int, List[Path]]:
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


def compute_audio_quality_metrics(
    *,
    enhanced: np.ndarray,
    clean_path: Optional[Path],
    noisy_ref: np.ndarray,
    sample_rate: int,
) -> Dict[str, Optional[float]]:
    metric_names = (
        "input_sdr",
        "sdr",
        "sdr_i",
        "input_si_sdr",
        "si_sdr",
        "si_sdr_i",
        "input_wb_pesq",
        "wb_pesq",
        "wb_pesq_i",
    )
    empty_metrics = {name: None for name in metric_names}
    if clean_path is None:
        return empty_metrics

    clean, clean_sr = load_mono_audio(clean_path, target_sr=sample_rate)
    if clean_sr != sample_rate:
        return empty_metrics

    min_len = min(len(enhanced), len(clean), len(noisy_ref))
    if min_len <= 0:
        return empty_metrics

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
        "input_wb_pesq": None,
        "wb_pesq": None,
        "wb_pesq_i": None,
    }


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
    input_sdr: Optional[float]
    sdr: Optional[float]
    sdr_i: Optional[float]
    input_si_sdr: Optional[float]
    si_sdr: Optional[float]
    si_sdr_i: Optional[float]
    input_wb_pesq: Optional[float]
    wb_pesq: Optional[float]
    wb_pesq_i: Optional[float]
    reference: str
    hypothesis: str
    localization_sec: float
    beamforming_sec: float
    whisper_sec: float
    total_sec: float
    localization_rtf: float
    beamforming_rtf: float
    whisper_rtf: float
    total_rtf: float
    under_realtime: int


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
    noisy_ref_for_metrics: np.ndarray,
    localization_sec: float,
    beam_sec: float,
) -> AsrTiming:
    beamformed_name = (
        f"hark_fileid_{fileid}_spk{ref.speaker_id}_"
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
    quality_metrics = compute_audio_quality_metrics(
        enhanced=enhanced_for_asr,
        clean_path=ref.clean_path,
        noisy_ref=noisy_ref_for_metrics,
        sample_rate=sr,
    )

    total_sec = localization_sec + beam_sec + whisper_sec
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
        input_sdr=quality_metrics["input_sdr"],
        sdr=quality_metrics["sdr"],
        sdr_i=quality_metrics["sdr_i"],
        input_si_sdr=quality_metrics["input_si_sdr"],
        si_sdr=quality_metrics["si_sdr"],
        si_sdr_i=quality_metrics["si_sdr_i"],
        input_wb_pesq=quality_metrics["input_wb_pesq"],
        wb_pesq=quality_metrics["wb_pesq"],
        wb_pesq_i=quality_metrics["wb_pesq_i"],
        reference=ref_text,
        hypothesis=hyp_text,
        localization_sec=localization_sec,
        beamforming_sec=beam_sec,
        whisper_sec=whisper_sec,
        total_sec=total_sec,
        localization_rtf=localization_sec / duration_sec,
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

    def mean_optional(name: str) -> float:
        values = [
            float(value)
            for row in evaluated
            for value in [getattr(row, name)]
            if value is not None and np.isfinite(float(value))
        ]
        return float(np.mean(values)) if values else 0.0

    return {
        "evaluated_utterances": len(evaluated),
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
        "mean_sample_wer": float(np.mean(wers)) if wers else 0.0,
        "mean_steering_doa_error_deg": float(np.mean(doa_errors)) if doa_errors else 0.0,
        "median_steering_doa_error_deg": float(np.median(doa_errors)) if doa_errors else 0.0,
        "mean_input_sdr": mean_optional("input_sdr"),
        "mean_sdr": mean_optional("sdr"),
        "mean_sdri": mean_optional("sdr_i"),
        "mean_input_si_sdr": mean_optional("input_si_sdr"),
        "mean_si_sdr": mean_optional("si_sdr"),
        "mean_sisdri": mean_optional("si_sdr_i"),
        "mean_input_wb_pesq": mean_optional("input_wb_pesq"),
        "mean_wb_pesq": mean_optional("wb_pesq"),
        "mean_wb_pesqi": mean_optional("wb_pesq_i"),
        "mean_localization_sec": float(np.mean([row.localization_sec for row in evaluated])) if evaluated else 0.0,
        "mean_beamforming_sec": float(np.mean([row.beamforming_sec for row in evaluated])) if evaluated else 0.0,
        "mean_whisper_sec": float(np.mean([row.whisper_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_sec": float(np.mean([row.total_sec for row in evaluated])) if evaluated else 0.0,
        "mean_total_rtf": float(np.mean([row.total_rtf for row in evaluated])) if evaluated else 0.0,
        "under_realtime_count": int(sum(row.under_realtime for row in evaluated)),
        "under_realtime_rate": float(np.mean([row.under_realtime for row in evaluated])) if evaluated else 0.0,
    }


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


def steering_matrix(
    freqs: np.ndarray,
    doas_deg: Sequence[float],
    mic_positions: np.ndarray,
    sound_speed: float,
) -> np.ndarray:
    num_freqs = len(freqs)
    num_mics = mic_positions.shape[0]
    num_sources = len(doas_deg)
    h_fms = np.empty((num_freqs, num_mics, num_sources), dtype=np.complex128)

    for src_idx, doa_deg in enumerate(doas_deg):
        theta = np.deg2rad(float(doa_deg) % 360.0)
        unit_vec = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
        delays = mic_positions.astype(np.float64) @ unit_vec / float(sound_speed)
        h_fms[:, :, src_idx] = np.exp(1j * 2.0 * np.pi * freqs[:, None] * delays[None, :])

    return h_fms


def stft_multichannel(
    wav_tc: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    spectra = []
    freqs = None
    for ch in range(wav_tc.shape[1]):
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
    assert freqs is not None
    return freqs, np.stack(spectra, axis=1).astype(np.complex128, copy=False)


def angular_distance_grid(a_deg: float, b_deg: np.ndarray) -> np.ndarray:
    return np.abs((b_deg - a_deg + 180.0) % 360.0 - 180.0)


def select_music_peaks(
    spectrum: np.ndarray,
    doa_grid: np.ndarray,
    num_sources: int,
    min_separation_deg: float,
) -> Tuple[List[int], List[float]]:
    peak_indices, _ = find_peaks(spectrum)
    if len(peak_indices) == 0:
        peak_indices = np.arange(len(spectrum))

    candidates = sorted(peak_indices.tolist(), key=lambda idx: float(spectrum[idx]), reverse=True)
    selected: List[int] = []
    selected_scores: List[float] = []

    for idx in candidates:
        doa = float(doa_grid[idx])
        if all(float(np.min(angular_distance_grid(doa, np.array([doa_grid[j]])))) >= min_separation_deg for j in selected):
            selected.append(idx)
            selected_scores.append(float(spectrum[idx]))
        if len(selected) == num_sources:
            break

    if len(selected) < num_sources:
        for idx in np.argsort(spectrum)[::-1]:
            idx = int(idx)
            if idx not in selected:
                selected.append(idx)
                selected_scores.append(float(spectrum[idx]))
            if len(selected) == num_sources:
                break

    doas = [int(round(float(doa_grid[idx]))) % 360 for idx in selected[:num_sources]]
    scores = selected_scores[:num_sources]
    return doas, scores


def localize_music_static(
    wav_tc: np.ndarray,
    sample_rate: int,
    num_sources: int,
    n_fft: int,
    hop_length: int,
    sound_speed: float,
    min_freq: float,
    max_freq: float,
    doa_step_deg: int,
    min_peak_separation_deg: float,
    diagonal_loading: float,
    mic_positions: np.ndarray = RESPEAKER4_MIC_POS,
) -> Tuple[List[int], List[float], np.ndarray, np.ndarray]:
    if wav_tc.ndim != 2:
        raise ValueError(f"Expected audio shape [samples, channels], got {wav_tc.shape}")
    if wav_tc.shape[1] < 2:
        raise ValueError("LocalizeMUSIC-style DOA estimation requires at least two channels.")
    if num_sources <= 0:
        raise ValueError("--num_sources must be positive.")
    if num_sources > wav_tc.shape[1] - 1:
        raise ValueError(
            "MUSIC signal-subspace dimension should be <= num_mics - 1. "
            f"Got num_sources={num_sources}, num_mics={wav_tc.shape[1]}."
        )

    num_mics = min(wav_tc.shape[1], mic_positions.shape[0])
    freqs, x_fmt = stft_multichannel(wav_tc[:, :num_mics], sample_rate, n_fft, hop_length)
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(freq_mask):
        raise ValueError("No STFT bins selected by --music_min_freq/--music_max_freq.")

    doa_grid = np.arange(0, 360, doa_step_deg, dtype=np.float64)
    h_fmg = steering_matrix(
        freqs=freqs[freq_mask],
        doas_deg=doa_grid,
        mic_positions=mic_positions[:num_mics],
        sound_speed=sound_speed,
    )
    h_fmg = h_fmg / np.maximum(np.linalg.norm(h_fmg, axis=1, keepdims=True), 1e-12)

    music_spectrum = np.zeros(len(doa_grid), dtype=np.float64)
    eye = np.eye(num_mics, dtype=np.complex128)

    for local_freq_idx, freq_idx in enumerate(np.flatnonzero(freq_mask)):
        x_mt = x_fmt[freq_idx]
        cov = x_mt @ x_mt.conj().T / max(x_mt.shape[1], 1)
        trace = float(np.real(np.trace(cov)) / num_mics)
        cov = cov + max(diagonal_loading * trace, 1e-8) * eye

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)
        noise_dim = max(num_mics - num_sources, 1)
        noise_vecs = eigvecs[:, order[:noise_dim]]
        steering = h_fmg[local_freq_idx]
        denom = np.sum(np.abs(noise_vecs.conj().T @ steering) ** 2, axis=0)
        music_spectrum += 1.0 / np.maximum(denom, 1e-12)

    music_spectrum /= max(int(np.sum(freq_mask)), 1)
    if np.max(music_spectrum) > 0:
        music_spectrum = music_spectrum / np.max(music_spectrum)

    doas, scores = select_music_peaks(
        spectrum=music_spectrum,
        doa_grid=doa_grid,
        num_sources=num_sources,
        min_separation_deg=min_peak_separation_deg,
    )
    return doas, scores, doa_grid, music_spectrum


def geometric_left_inverse(h_ms: np.ndarray, diagonal_loading: float) -> np.ndarray:
    num_sources = h_ms.shape[1]
    gram = h_ms.conj().T @ h_ms
    trace = float(np.real(np.trace(gram)) / max(num_sources, 1))
    load = max(diagonal_loading * trace, 1e-8)
    return np.linalg.solve(
        gram + load * np.eye(num_sources, dtype=np.complex128),
        h_ms.conj().T,
    )


def ghdss_nonlinearity(y_st: np.ndarray, scale: float) -> np.ndarray:
    magnitude = np.abs(y_st)
    phase = np.exp(1j * np.angle(y_st))
    return np.tanh(scale * magnitude) * phase


def normalize_demix_rows(
    w_sm: np.ndarray,
    x_mt: np.ndarray,
    target_power: np.ndarray,
    eps: float,
) -> np.ndarray:
    y_st = w_sm @ x_mt
    power = np.mean(np.abs(y_st) ** 2, axis=1)
    gain = np.sqrt(np.maximum(target_power, eps) / np.maximum(power, eps))
    return w_sm * gain[:, None]


def ghdss_update(
    *,
    x_mt: np.ndarray,
    h_ms: np.ndarray,
    w0_sm: np.ndarray,
    iterations: int,
    step_size: float,
    alpha_ss: float,
    beta_lc: float,
    ss_scale: float,
    eps: float,
) -> np.ndarray:
    if iterations <= 0:
        return w0_sm

    num_sources = w0_sm.shape[0]
    eye = np.eye(num_sources, dtype=np.complex128)
    h_norm = max(float(np.linalg.norm(h_ms) ** 2), eps)
    w_sm = w0_sm.copy()
    y0_st = w0_sm @ x_mt
    target_power = np.mean(np.abs(y0_st) ** 2, axis=1)

    for _ in range(iterations):
        y_st = w_sm @ x_mt
        phi_y = ghdss_nonlinearity(y_st, scale=ss_scale)
        r_phi_y = phi_y @ y_st.conj().T / max(y_st.shape[1], 1)
        offdiag = r_phi_y - np.diag(np.diag(r_phi_y))
        ss_grad = offdiag @ w_sm

        lc_error = (w_sm @ h_ms) - eye
        lc_grad = (lc_error @ h_ms.conj().T) / h_norm

        w_sm = w_sm - step_size * (alpha_ss * ss_grad + beta_lc * lc_grad)
        w_sm = normalize_demix_rows(w_sm, x_mt, target_power, eps)

    return w_sm


def ghdss_separate(
    wav_tc: np.ndarray,
    sample_rate: int,
    doas_deg: Sequence[float],
    n_fft: int,
    hop_length: int,
    sound_speed: float,
    diagonal_loading: float,
    iterations: int,
    step_size: float,
    alpha_ss: float,
    beta_lc: float,
    ss_scale: float,
    mic_positions: np.ndarray = RESPEAKER4_MIC_POS,
    eps: float = 1e-8,
) -> np.ndarray:
    if wav_tc.ndim != 2:
        raise ValueError(f"Expected audio shape [samples, channels], got {wav_tc.shape}")
    if wav_tc.shape[1] < 1:
        raise ValueError("GHDSS-style separation requires at least one channel.")
    if not doas_deg:
        raise ValueError("At least one source DOA is required.")
    if hop_length <= 0 or hop_length >= n_fft:
        raise ValueError("--frame_shift must be > 0 and smaller than --frame_length.")

    num_mics = min(wav_tc.shape[1], mic_positions.shape[0])
    num_sources = len(doas_deg)
    if num_mics == 1:
        return np.tile(wav_tc[:, :1].T.astype(np.float32), (num_sources, 1))

    freqs, x_fmt = stft_multichannel(wav_tc[:, :num_mics], sample_rate, n_fft, hop_length)
    h_fms = steering_matrix(
        freqs=freqs,
        doas_deg=doas_deg,
        mic_positions=mic_positions[:num_mics],
        sound_speed=sound_speed,
    )

    y_fst = np.empty((x_fmt.shape[0], num_sources, x_fmt.shape[2]), dtype=np.complex128)
    for freq_idx in range(x_fmt.shape[0]):
        x_mt = x_fmt[freq_idx]
        h_ms = h_fms[freq_idx]
        w0_sm = geometric_left_inverse(h_ms, diagonal_loading=diagonal_loading)
        w_sm = ghdss_update(
            x_mt=x_mt,
            h_ms=h_ms,
            w0_sm=w0_sm,
            iterations=iterations,
            step_size=step_size,
            alpha_ss=alpha_ss,
            beta_lc=beta_lc,
            ss_scale=ss_scale,
            eps=eps,
        )
        y_fst[freq_idx] = w_sm @ x_mt

    separated = []
    for src_idx in range(num_sources):
        _, wav = istft(
            y_fst[:, src_idx, :],
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            input_onesided=True,
            boundary=True,
        )
        wav = np.nan_to_num(wav, copy=False)
        if wav.shape[0] < wav_tc.shape[0]:
            wav = np.pad(wav, (0, wav_tc.shape[0] - wav.shape[0]))
        separated.append(wav[: wav_tc.shape[0]].astype(np.float32))

    return np.stack(separated, axis=0)


def match_hark_doas_to_scene_refs(
    scene_refs: Sequence[TargetReference],
    pred_doas: Sequence[int],
) -> Dict[int, Tuple[int, int, float]]:
    if len(pred_doas) < len(scene_refs):
        raise ValueError(f"Need at least {len(scene_refs)} HARK DOAs, got {len(pred_doas)}.")

    best_perm: Optional[Tuple[int, ...]] = None
    best_cost = float("inf")
    import itertools

    for perm in itertools.permutations(range(len(pred_doas)), len(scene_refs)):
        cost = sum(
            circular_angle_error_deg(pred_doas[pred_idx], ref.gt_doa)
            for ref, pred_idx in zip(scene_refs, perm)
        )
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    if best_perm is None:
        return {}

    matched: Dict[int, Tuple[int, int, float]] = {}
    for ref, pred_idx in zip(scene_refs, best_perm):
        pred_doa = int(pred_doas[pred_idx])
        matched[ref.speaker_id] = (
            pred_idx,
            pred_doa,
            circular_angle_error_deg(pred_doa, ref.gt_doa),
        )
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HARK-style LocalizeMUSIC -> GHDSS -> Whisper small benchmark."
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument(
        "--target_speaker_id",
        type=int,
        default=1,
        help="Speaker id to evaluate. Use 0 to evaluate all three separated speakers.",
    )
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--frame_shift", type=int, default=256)
    parser.add_argument("--sound_speed", type=float, default=343.0)
    parser.add_argument("--music_min_freq", type=float, default=500.0)
    parser.add_argument("--music_max_freq", type=float, default=2800.0)
    parser.add_argument("--music_doa_step", type=int, default=5)
    parser.add_argument("--music_min_peak_separation", type=float, default=25.0)
    parser.add_argument("--music_diag_load", type=float, default=1e-3)
    parser.add_argument("--ghdss_diag_load", type=float, default=1e-3)
    parser.add_argument("--ghdss_iterations", type=int, default=8)
    parser.add_argument("--ghdss_step_size", type=float, default=0.05)
    parser.add_argument("--ghdss_alpha_ss", type=float, default=1.0)
    parser.add_argument("--ghdss_beta_lc", type=float, default=1.0)
    parser.add_argument("--ghdss_ss_scale", type=float, default=1.0)
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save each selected separated target wav.")
    parser.add_argument("--save_music_spectrum", action="store_true", help="Save per-scene MUSIC spectrum CSV files.")
    return parser.parse_args()


def append_separated_target(
    *,
    args: argparse.Namespace,
    whisper_model,
    enhanced_dir: Path,
    fileid: int,
    mic_file: str,
    duration_sec: float,
    sr: int,
    ref: TargetReference,
    separated_sources_st: np.ndarray,
    source_index: int,
    hark_doas: Sequence[int],
    hark_scores: Sequence[float],
    matched_pred_idx: int,
    steering_doa: int,
    steering_error: float,
    noisy_ref_for_metrics: np.ndarray,
    localization_sec: float,
    separation_sec: float,
) -> AsrTiming:
    enhanced_for_asr = separated_sources_st[source_index].astype(np.float32, copy=False)
    row = append_result_for_target(
        args=args,
        whisper_model=whisper_model,
        enhanced_dir=enhanced_dir,
        fileid=fileid,
        mic_file=mic_file,
        duration_sec=duration_sec,
        sr=sr,
        ref=ref,
        method="HARK-LocalizeMUSIC-GHDSS",
        doa_source="hark_music",
        steering_doa=steering_doa,
        pred_doas=hark_doas,
        matched_pred_idx=matched_pred_idx,
        steering_error=steering_error,
        enhanced_for_asr=enhanced_for_asr,
        noisy_ref_for_metrics=noisy_ref_for_metrics,
        localization_sec=localization_sec,
        beam_sec=separation_sec,
    )
    return row


def maybe_save_music_spectrum(
    out_dir: Path,
    fileid: int,
    doa_grid: np.ndarray,
    spectrum: np.ndarray,
) -> None:
    spectrum_dir = out_dir / "music_spectra"
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    path = spectrum_dir / f"music_spectrum_fileid_{fileid}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doa_deg", "music_spectrum_normalized"])
        for doa, value in zip(doa_grid, spectrum):
            writer.writerow([float(doa), float(value)])


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
    if args.frame_shift <= 0 or args.frame_shift >= args.frame_length:
        raise ValueError("--frame_shift must be > 0 and smaller than --frame_length.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_hark_enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Whisper model: {args.whisper_model}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"Target speaker: {'all' if args.target_speaker_id == 0 else f'spk{args.target_speaker_id}'}")
    print("Pipeline: HARK-style LocalizeMUSIC -> SourceTracker-lite -> GHDSS -> Whisper")
    print("DOA source: HARK-style MUSIC only; no GT DOA and no IPDNet DOA are used.")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Target text folder: {args.text_dir}")

    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    scene_refs_by_fileid, target_refs_by_fileid = load_scene_references(
        args.text_dir,
        args.clean_dir,
        target_speaker_id=args.target_speaker_id,
    )

    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print(f"Scene references: {sum(len(v) for v in scene_refs_by_fileid.values())}")
    print(f"Target references: {sum(len(v) for v in target_refs_by_fileid.values())}")

    scene_results: List[AsrTiming] = []
    localization_records: List[Dict[str, object]] = []
    skipped_no_scene_refs = 0
    skipped_no_target_refs = 0
    skipped_bad_scene_ref_count = 0
    skipped_bad_localization = 0

    for fileid, target_paths in tqdm(grouped.items(), desc="HARK-ASR", unit="scene"):
        scene_refs = scene_refs_by_fileid.get(fileid, [])
        target_refs = target_refs_by_fileid.get(fileid, [])
        if not scene_refs:
            skipped_no_scene_refs += 1
            print(f"fileid={fileid}: no scene text references, skipped.")
            continue
        if not target_refs:
            skipped_no_target_refs += 1
            print(f"fileid={fileid}: no matching target speaker text references, skipped.")
            continue
        if len(scene_refs) != args.num_sources:
            skipped_bad_scene_ref_count += 1
            print(
                f"fileid={fileid}: expected {args.num_sources} scene refs, "
                f"found {len(scene_refs)}, skipped."
            )
            continue

        mic_path = choose_representative_mic(target_paths)
        wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
        noisy_ref_for_metrics = wav_tc[:, 0]
        duration_sec = wav_tc.shape[0] / float(sr)

        try:
            music_result, localization_sec = elapsed_seconds(
                "cpu",
                lambda: localize_music_static(
                    wav_tc=wav_tc,
                    sample_rate=sr,
                    num_sources=args.num_sources,
                    n_fft=args.frame_length,
                    hop_length=args.frame_shift,
                    sound_speed=args.sound_speed,
                    min_freq=args.music_min_freq,
                    max_freq=args.music_max_freq,
                    doa_step_deg=args.music_doa_step,
                    min_peak_separation_deg=args.music_min_peak_separation,
                    diagonal_loading=args.music_diag_load,
                ),
            )
            hark_doas, hark_scores, doa_grid, music_spectrum = music_result
        except Exception as exc:
            skipped_bad_localization += 1
            print(f"fileid={fileid}: LocalizeMUSIC-style localization failed: {exc}")
            continue

        if len(hark_doas) != args.num_sources:
            skipped_bad_localization += 1
            print(
                f"fileid={fileid}: expected {args.num_sources} HARK DOAs, "
                f"got {len(hark_doas)}, skipped."
            )
            continue

        if args.save_music_spectrum:
            maybe_save_music_spectrum(args.out_dir, fileid, doa_grid, music_spectrum)

        matched = match_hark_doas_to_scene_refs(scene_refs, hark_doas)
        gt_doas = [ref.gt_doa for ref in scene_refs]
        localization_records.append(
            {
                "fileid": fileid,
                "mic_file": mic_path.name,
                "gt_doas_by_scene_ref_order": ",".join(str(doa) for doa in gt_doas),
                "hark_music_doas": ",".join(str(doa) for doa in hark_doas),
                "hark_music_scores": ",".join(f"{score:.6f}" for score in hark_scores),
                "mean_matched_doa_error_deg": float(
                    np.mean([matched[ref.speaker_id][2] for ref in scene_refs])
                ),
                "localization_sec": localization_sec,
                "localization_rtf": localization_sec / duration_sec,
            }
        )

        separated, separation_sec = elapsed_seconds(
            "cpu",
            lambda: ghdss_separate(
                wav_tc=wav_tc,
                sample_rate=sr,
                doas_deg=hark_doas,
                n_fft=args.frame_length,
                hop_length=args.frame_shift,
                sound_speed=args.sound_speed,
                diagonal_loading=args.ghdss_diag_load,
                iterations=args.ghdss_iterations,
                step_size=args.ghdss_step_size,
                alpha_ss=args.ghdss_alpha_ss,
                beta_lc=args.ghdss_beta_lc,
                ss_scale=args.ghdss_ss_scale,
            ),
        )

        for ref in target_refs:
            matched_pred_idx, steering_doa, steering_error = matched[ref.speaker_id]
            scene_results.append(
                append_separated_target(
                    args=args,
                    whisper_model=whisper_model,
                    enhanced_dir=enhanced_dir,
                    fileid=fileid,
                    mic_file=mic_path.name,
                    duration_sec=duration_sec,
                    sr=sr,
                    ref=ref,
                    separated_sources_st=separated,
                    source_index=matched_pred_idx,
                    hark_doas=hark_doas,
                    hark_scores=hark_scores,
                    matched_pred_idx=matched_pred_idx,
                    steering_doa=steering_doa,
                    steering_error=steering_error,
                    noisy_ref_for_metrics=noisy_ref_for_metrics,
                    localization_sec=localization_sec,
                    separation_sec=separation_sec,
                )
            )

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_hark_wer_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(AsrTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    localization_csv = args.out_dir / "hark_music_localization_details.csv"
    with localization_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "fileid",
            "mic_file",
            "gt_doas_by_scene_ref_order",
            "hark_music_doas",
            "hark_music_scores",
            "mean_matched_doa_error_deg",
            "localization_sec",
            "localization_rtf",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in localization_records:
            writer.writerow(row)

    by_method = {
        method: summarize_method([row for row in scene_results if row.method == method])
        for method in sorted({row.method for row in scene_results})
    }
    all_summary = summarize_method(scene_results)

    localization_errors = [
        float(row["mean_matched_doa_error_deg"])
        for row in localization_records
        if np.isfinite(float(row["mean_matched_doa_error_deg"]))
    ]
    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
        "text_dir": str(args.text_dir),
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "enhancement": "hark_architecture_localizemusic_sourcetracker_ghdss",
        "official_hark_runtime": False,
        "official_hark_note": (
            "This run uses a Python implementation of the original HARK architecture. "
            "It does not call official HARK/PyHARK binaries."
        ),
        "doa_source": "hark_style_music",
        "target_speaker_id": args.target_speaker_id,
        "num_sources": args.num_sources,
        "frame_length": args.frame_length,
        "frame_shift": args.frame_shift,
        "sound_speed": args.sound_speed,
        "music_min_freq": args.music_min_freq,
        "music_max_freq": args.music_max_freq,
        "music_doa_step": args.music_doa_step,
        "music_min_peak_separation": args.music_min_peak_separation,
        "music_diag_load": args.music_diag_load,
        "ghdss_diag_load": args.ghdss_diag_load,
        "ghdss_iterations": args.ghdss_iterations,
        "ghdss_step_size": args.ghdss_step_size,
        "ghdss_alpha_ss": args.ghdss_alpha_ss,
        "ghdss_beta_lc": args.ghdss_beta_lc,
        "ghdss_ss_scale": args.ghdss_ss_scale,
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "scene_references": sum(len(v) for v in scene_refs_by_fileid.values()),
        "target_text_references": sum(len(v) for v in target_refs_by_fileid.values()),
        "evaluated_utterances": len(scene_results),
        "localized_scenes": len(localization_records),
        "mean_scene_doa_error_deg": float(np.mean(localization_errors)) if localization_errors else 0.0,
        "median_scene_doa_error_deg": float(np.median(localization_errors)) if localization_errors else 0.0,
        "skipped_no_scene_refs": skipped_no_scene_refs,
        "skipped_no_target_refs": skipped_no_target_refs,
        "skipped_bad_scene_ref_count": skipped_bad_scene_ref_count,
        "skipped_bad_localization": skipped_bad_localization,
        "overall": all_summary,
        "by_method": by_method,
    }

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_hark_wer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== HARK-STYLE ASR SUMMARY =====")
    print(f"Target speaker: {'all' if args.target_speaker_id == 0 else f'spk{args.target_speaker_id}'}")
    print(f"Localized scenes: {summary['localized_scenes']}")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Mean scene DOA error: {summary['mean_scene_doa_error_deg']:.2f} deg")
    print(f"Skipped scenes without scene refs: {summary['skipped_no_scene_refs']}")
    print(f"Skipped scenes without target refs: {summary['skipped_no_target_refs']}")
    print(f"Skipped scenes with wrong scene-ref count: {summary['skipped_bad_scene_ref_count']}")
    print(f"Skipped localization failures: {summary['skipped_bad_localization']}")
    for method, stats in by_method.items():
        print(
            f"{method}: utterances={stats['evaluated_utterances']}, "
            f"corpus WER={stats['corpus_wer']:.4f}, "
            f"mean sample WER={stats['mean_sample_wer']:.4f}, "
            f"mean DOA error={stats['mean_steering_doa_error_deg']:.2f} deg, "
            f"mean total RTF={stats['mean_total_rtf']:.3f}"
        )
        print(
            f"    mean SDRi={stats['mean_sdri']:.4f}, "
            f"mean SISDRi={stats['mean_sisdri']:.4f}, "
            f"mean PESQ={stats['mean_wb_pesq']:.4f}"
        )
    print(f"Saved ASR details: {details_csv}")
    print(f"Saved localization details: {localization_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
