"""
Offline single-mic SepFormer -> Whisper WER benchmark.

For each saved 4-channel ReSpeaker scene, this script selects one fixed mic
channel, resamples it from 16 kHz to the 8 kHz expected by SpeechBrain's
WSJ0-3Mix SepFormer, separates three sources, resamples each source back to
16 kHz, then evaluates WER and audio metrics against the dataset references.

SepFormer source order is arbitrary, so references are matched to separated
sources with oracle SI-SDR against the clean files before Whisper/metric
evaluation. Use --target_speaker_id 1 to reproduce the dominant-speaker-only
style used by MVDR.py; the default evaluates all available speakers.
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
from scipy.signal import resample_poly
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
DSE_ROOT = MODELS_ROOT / "DSE"
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"

sys.path.insert(0, str(DSE_ROOT))
from models.utils.metrics import cal_metrics_functional  # noqa: E402


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


def resample_1d(wav: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if source_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    gcd = math.gcd(source_sr, target_sr)
    up = target_sr // gcd
    down = source_sr // gcd
    return resample_poly(wav, up, down).astype(np.float32)


def match_length(wav: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.shape[0] < target_len:
        wav = np.pad(wav, (0, target_len - wav.shape[0]))
    return wav[:target_len].astype(np.float32, copy=False)


def load_multichannel_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        wav = np.stack([resample_1d(wav[:, ch], sr, target_sr) for ch in range(wav.shape[1])], axis=1)
        sr = target_sr
    return wav, sr


def load_mono_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        wav = resample_1d(wav, sr, target_sr)
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


def load_target_references(
    text_dir: Path,
    clean_dir: Path,
    target_speaker_id: int,
) -> Dict[int, List[TargetReference]]:
    clean_pattern = (
        "clean_fileid_*_doa*_spk*.wav"
        if target_speaker_id <= 0
        else f"clean_fileid_*_doa*_spk{target_speaker_id}.wav"
    )
    text_pattern = (
        "text_fileid_*_doa*_spk*.txt"
        if target_speaker_id <= 0
        else f"text_fileid_*_doa*_spk{target_speaker_id}.txt"
    )

    clean_by_key: Dict[Tuple[int, int, int], Path] = {}
    for clean_path in sorted(clean_dir.glob(clean_pattern)):
        fileid = parse_fileid(clean_path)
        speaker_id = parse_speaker_id(clean_path)
        doa = parse_doa(clean_path)
        clean_by_key[(fileid, speaker_id, doa)] = clean_path

    references: Dict[int, List[TargetReference]] = {}
    for text_path in sorted(text_dir.glob(text_pattern)):
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


def import_sepformer_class():
    try:
        from speechbrain.inference.separation import SepformerSeparation

        return SepformerSeparation
    except ImportError:
        try:
            from speechbrain.pretrained import SepformerSeparation

            return SepformerSeparation
        except ImportError as exc:
            raise ImportError(
                "SpeechBrain is required for this baseline. Install it with: pip install speechbrain"
            ) from exc


def load_sepformer(source: str, savedir: Path, device: str):
    sepformer_cls = import_sepformer_class()
    kwargs = {
        "source": source,
        "savedir": str(savedir),
    }
    if device:
        kwargs["run_opts"] = {"device": device}
    return sepformer_cls.from_hparams(**kwargs)


def separate_file_compat(separator, path: Path) -> torch.Tensor:
    try:
        est_sources = separator.separate_file(path=str(path))
    except TypeError:
        est_sources = separator.separate_file(str(path))
    if not isinstance(est_sources, torch.Tensor):
        est_sources = torch.as_tensor(est_sources)
    return est_sources


def tensor_to_sources_bt(est_sources: torch.Tensor) -> np.ndarray:
    est_sources = est_sources.detach().cpu().float()
    if est_sources.ndim == 3:
        if est_sources.shape[0] != 1:
            raise ValueError(f"Expected SepFormer batch size 1, got shape {tuple(est_sources.shape)}")
        est_sources = est_sources[0]
    if est_sources.ndim != 2:
        raise ValueError(f"Expected SepFormer output [time, sources] or [1, time, sources], got {tuple(est_sources.shape)}")
    if est_sources.shape[0] < est_sources.shape[1]:
        est_sources = est_sources.transpose(0, 1)
    return est_sources.numpy().astype(np.float32)


def run_sepformer_scene(
    *,
    separator,
    mic_path: Path,
    fileid: int,
    args: argparse.Namespace,
    tmp_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
    mic_idx = args.mic_channel - 1
    if mic_idx < 0 or mic_idx >= wav_tc.shape[1]:
        raise ValueError(f"--mic_channel {args.mic_channel} is invalid for {wav_tc.shape[1]}-channel file: {mic_path}")

    mono_16k = wav_tc[:, mic_idx].astype(np.float32)
    mono_8k = resample_1d(mono_16k, args.sample_rate, args.sepformer_sample_rate)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sepformer_input = tmp_dir / f"sepformer_input_fileid_{fileid}_ch{args.mic_channel}_8k.wav"
    sf.write(str(sepformer_input), mono_8k, args.sepformer_sample_rate)

    def run_separation():
        with torch.inference_mode():
            return separate_file_compat(separator, sepformer_input)

    est_sources, sepformer_sec = elapsed_seconds(args.device, run_separation)
    sources_8k_ts = tensor_to_sources_bt(est_sources)
    sources_8k_ts = np.stack(
        [match_length(sources_8k_ts[:, idx], len(mono_8k)) for idx in range(sources_8k_ts.shape[1])],
        axis=1,
    )
    sources_16k_st = np.stack(
        [
            match_length(resample_1d(sources_8k_ts[:, idx], args.sepformer_sample_rate, args.sample_rate), len(mono_16k))
            for idx in range(sources_8k_ts.shape[1])
        ],
        axis=0,
    )

    if not args.keep_preprocessed:
        try:
            sepformer_input.unlink()
        except OSError:
            pass

    return sources_16k_st, sources_8k_ts.T, mono_16k, sepformer_sec, sr


def simple_si_sdr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    min_len = min(pred.shape[0], target.shape[0])
    if min_len <= 0:
        return -float("inf")
    pred = pred[:min_len] - np.mean(pred[:min_len])
    target = target[:min_len] - np.mean(target[:min_len])
    target_energy = np.sum(target**2) + eps
    projection = np.sum(pred * target) * target / target_energy
    noise = pred - projection
    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)))


def match_sources_to_targets(
    sources_16k_st: np.ndarray,
    references: Sequence[TargetReference],
    sample_rate: int,
) -> Dict[int, Tuple[int, float]]:
    if len(references) > sources_16k_st.shape[0]:
        raise ValueError(
            f"Need at least {len(references)} separated sources, got {sources_16k_st.shape[0]}."
        )
    if any(ref.clean_path is None for ref in references):
        return {
            ref.speaker_id: (idx, float("nan"))
            for idx, ref in enumerate(references)
            if idx < sources_16k_st.shape[0]
        }

    clean_sources = [
        match_length(load_mono_audio(ref.clean_path, target_sr=sample_rate)[0], sources_16k_st.shape[1])
        for ref in references
    ]
    scores = np.zeros((len(references), sources_16k_st.shape[0]), dtype=np.float64)
    for ref_idx, clean in enumerate(clean_sources):
        for source_idx in range(sources_16k_st.shape[0]):
            scores[ref_idx, source_idx] = simple_si_sdr(sources_16k_st[source_idx], clean)

    best_perm: Optional[Tuple[int, ...]] = None
    best_score = -float("inf")
    for perm in itertools.permutations(range(sources_16k_st.shape[0]), len(references)):
        score = sum(scores[ref_idx, source_idx] for ref_idx, source_idx in enumerate(perm))
        if score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return {}
    return {
        ref.speaker_id: (source_idx, float(scores[ref_idx, source_idx]))
        for ref_idx, (ref, source_idx) in enumerate(zip(references, best_perm))
    }


def scalar_only(values: Dict[str, object]) -> Dict[str, Optional[float]]:
    scalars: Dict[str, Optional[float]] = {}
    for key, value in values.items():
        if key.endswith("_all"):
            continue
        if value is None:
            scalars[key] = None
        elif isinstance(value, torch.Tensor):
            scalars[key] = float(value.item()) if value.numel() == 1 else None
        elif isinstance(value, np.ndarray):
            scalars[key] = float(value.item()) if value.size == 1 else None
        elif isinstance(value, (int, float, np.floating)):
            scalars[key] = float(value)
    return scalars


def compute_audio_quality_metrics(
    *,
    enhanced: np.ndarray,
    clean_path: Optional[Path],
    noisy_ref: np.ndarray,
    sample_rate: int,
    device: str,
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

    preds = torch.from_numpy(enhanced[:min_len].copy()).unsqueeze(0).float().to(device)
    target = torch.from_numpy(clean[:min_len].copy()).unsqueeze(0).float().to(device)
    original = torch.from_numpy(noisy_ref[:min_len].copy()).unsqueeze(0).float().to(device)

    metrics, input_metrics, imp_metrics = cal_metrics_functional(
        ["SDR", "SI_SDR", "WB_PESQ"],
        preds,
        target,
        original,
        sample_rate,
        device_only=None,
    )
    values = {}
    values.update(scalar_only(input_metrics))
    values.update(scalar_only(metrics))
    values.update(scalar_only(imp_metrics))
    return {name: values.get(name) for name in metric_names}


@dataclass
class AsrTiming:
    fileid: int
    mic_file: str
    method: str
    speaker_id: int
    duration_sec: float
    gt_doa: int
    mic_channel: int
    sepformer_source: str
    separated_source_index: int
    oracle_match_si_sdr: Optional[float]
    enhanced_file: str
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
    sepformer_sec: float
    whisper_sec: float
    metrics_sec: float
    total_sec: float
    sepformer_rtf: float
    whisper_rtf: float
    metrics_rtf: float
    total_rtf: float
    under_realtime: int


def default_savedir(source: str) -> Path:
    return PROJECT_ROOT / "pretrained_models" / source.replace("/", "_")


def normalize_sepformer_source(source: str) -> str:
    aliases = {
        "sepformer-wsj03mix": "speechbrain/sepformer-wsj03mix",
        "sepformer-libri3mix": "speechbrain/sepformer-libri3mix",
    }
    return aliases.get(source, source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-mic SepFormer -> Whisper WER benchmark.")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--sepformer_sample_rate", type=int, default=8000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument(
        "--mic_channel",
        type=int,
        default=1,
        help="1-based ReSpeaker channel to feed into SepFormer. Default selects channel 1.",
    )
    parser.add_argument(
        "--sepformer_source",
        type=str,
        default="speechbrain/sepformer-wsj03mix",
        help="SpeechBrain Hugging Face model id, e.g. speechbrain/sepformer-wsj03mix or speechbrain/sepformer-libri3mix.",
    )
    parser.add_argument(
        "--sepformer_savedir",
        type=Path,
        default=None,
        help="Local SpeechBrain model cache folder. Defaults to pretrained_models/<model-id>.",
    )
    parser.add_argument(
        "--target_speaker_id",
        type=int,
        default=0,
        help="Speaker id to evaluate; 0 means all available speakers. Use 1 to mirror MVDR.py's dominant speaker run.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save 16 kHz separated sources ready for Whisper.")
    parser.add_argument("--save_8k_sources", action="store_true", help="Also save the raw 8 kHz SepFormer separated sources.")
    parser.add_argument("--keep_preprocessed", action="store_true", help="Keep the temporary 8 kHz mono SepFormer input files.")
    return parser.parse_args()


def save_sources(
    *,
    args: argparse.Namespace,
    enhanced_dir: Path,
    fileid: int,
    mic_file: str,
    sources_16k_st: np.ndarray,
    sources_8k_st: np.ndarray,
) -> List[str]:
    saved_16k_names = []
    stem = Path(mic_file).stem
    for source_idx in range(sources_16k_st.shape[0]):
        name_16k = f"sepformer_fileid_{fileid}_{stem}_ch{args.mic_channel}_source{source_idx + 1}_16k.wav"
        sf.write(str(enhanced_dir / name_16k), sources_16k_st[source_idx], args.sample_rate)
        saved_16k_names.append(name_16k)
        if args.save_8k_sources:
            name_8k = f"sepformer_fileid_{fileid}_{stem}_ch{args.mic_channel}_source{source_idx + 1}_8k.wav"
            sf.write(str(enhanced_dir / name_8k), sources_8k_st[source_idx], args.sepformer_sample_rate)
    return saved_16k_names


def append_result_for_target(
    *,
    args: argparse.Namespace,
    whisper_model,
    fileid: int,
    mic_file: str,
    duration_sec: float,
    sr: int,
    ref: TargetReference,
    enhanced_for_asr: np.ndarray,
    noisy_ref_for_metrics: np.ndarray,
    source_idx: int,
    match_score: Optional[float],
    enhanced_file: str,
    sepformer_sec: float,
) -> AsrTiming:
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

    quality_metrics, metrics_sec = elapsed_seconds(
        args.device,
        lambda: compute_audio_quality_metrics(
            enhanced=enhanced_for_asr,
            clean_path=ref.clean_path,
            noisy_ref=noisy_ref_for_metrics,
            sample_rate=sr,
            device=args.device,
        ),
    )

    total_sec = sepformer_sec + whisper_sec + metrics_sec
    return AsrTiming(
        fileid=fileid,
        mic_file=mic_file,
        method=f"SepFormer-{args.sepformer_source.rsplit('/', 1)[-1]}",
        speaker_id=ref.speaker_id,
        duration_sec=duration_sec,
        gt_doa=ref.gt_doa,
        mic_channel=args.mic_channel,
        sepformer_source=args.sepformer_source,
        separated_source_index=source_idx,
        oracle_match_si_sdr=match_score,
        enhanced_file=enhanced_file,
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
        sepformer_sec=sepformer_sec,
        whisper_sec=whisper_sec,
        metrics_sec=metrics_sec,
        total_sec=total_sec,
        sepformer_rtf=sepformer_sec / duration_sec,
        whisper_rtf=whisper_sec / duration_sec,
        metrics_rtf=metrics_sec / duration_sec,
        total_rtf=total_sec / duration_sec,
        under_realtime=int(total_sec < duration_sec),
    )


def summarize_method(rows: Sequence[AsrTiming]) -> Dict[str, float | int]:
    evaluated = [row for row in rows if row.wer is not None]
    total_edits = sum(int(row.edit_distance or 0) for row in evaluated)
    total_ref_words = sum(int(row.ref_words or 0) for row in evaluated)
    wers = [float(row.wer) for row in evaluated]

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
        "mean_oracle_match_si_sdr": mean_optional("oracle_match_si_sdr"),
        "mean_input_sdr": mean_optional("input_sdr"),
        "mean_sdr": mean_optional("sdr"),
        "mean_sdri": mean_optional("sdr_i"),
        "mean_input_si_sdr": mean_optional("input_si_sdr"),
        "mean_si_sdr": mean_optional("si_sdr"),
        "mean_sisdri": mean_optional("si_sdr_i"),
        "mean_input_wb_pesq": mean_optional("input_wb_pesq"),
        "mean_wb_pesq": mean_optional("wb_pesq"),
        "mean_wb_pesqi": mean_optional("wb_pesq_i"),
        "mean_sepformer_sec": float(np.mean([row.sepformer_sec for row in evaluated])) if evaluated else 0.0,
        "mean_whisper_sec": float(np.mean([row.whisper_sec for row in evaluated])) if evaluated else 0.0,
        "mean_metrics_sec": float(np.mean([row.metrics_sec for row in evaluated])) if evaluated else 0.0,
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SepFormer/metrics were requested on CUDA, but torch.cuda.is_available() is False.")
    if args.target_speaker_id > args.num_sources:
        raise ValueError(
            f"--target_speaker_id {args.target_speaker_id} is larger than --num_sources {args.num_sources}. "
            "For dataset_4mic_3spk, use 1, 2, 3, or 0 for all speakers."
        )
    if args.sepformer_sample_rate != 8000:
        print("Warning: SpeechBrain WSJ0/Libri3Mix SepFormer models are documented for 8 kHz input.")

    args.sepformer_source = normalize_sepformer_source(args.sepformer_source)
    args.sepformer_savedir = args.sepformer_savedir or default_savedir(args.sepformer_source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_sepformer_enhanced"
    if args.save_enhanced or args.save_8k_sources:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir / "_tmp_sepformer_input_8k"

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"SepFormer source: {args.sepformer_source}")
    print(f"SepFormer savedir: {args.sepformer_savedir}")
    print(f"Mic channel: {args.mic_channel} (1-based)")
    print(f"Target speaker: {'all' if args.target_speaker_id <= 0 else f'spk{args.target_speaker_id}'}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Target text folder: {args.text_dir}")

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    references_by_fileid = load_target_references(
        args.text_dir,
        args.clean_dir,
        target_speaker_id=args.target_speaker_id,
    )
    target_ref_count = sum(len(v) for v in references_by_fileid.values())
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print(f"Target text references: {target_ref_count}")
    if target_ref_count == 0:
        speaker_hint = "spk*" if args.target_speaker_id <= 0 else f"spk{args.target_speaker_id}"
        raise FileNotFoundError(
            f"No target text references matched text_fileid_*_doa*_{speaker_hint}.txt in {args.text_dir}. "
            "For dataset_4mic_3spk, use --target_speaker_id 1, 2, 3, or omit it/leave it at 0 for all speakers."
        )

    print("Loading SpeechBrain SepFormer once...")
    separator = load_sepformer(args.sepformer_source, args.sepformer_savedir, args.device)

    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    scene_results: List[AsrTiming] = []
    skipped_no_gt_targets = 0
    skipped_missing_match = 0
    truncated_target_refs = 0

    for fileid, target_paths in tqdm(grouped.items(), desc="SepFormer-ASR", unit="scene"):
        refs = references_by_fileid.get(fileid, [])
        if not refs:
            skipped_no_gt_targets += 1
            print(f"fileid={fileid}: no matching text_fileid_*_doa*_spk*.txt references, skipped.")
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
        sources_16k_st, sources_8k_st, noisy_ref_for_metrics, sepformer_sec, sr = run_sepformer_scene(
            separator=separator,
            mic_path=mic_path,
            fileid=fileid,
            args=args,
            tmp_dir=tmp_dir,
        )
        if sources_16k_st.shape[0] < len(refs):
            skipped_missing_match += 1
            print(
                f"fileid={fileid}: expected at least {len(refs)} separated sources, "
                f"got {sources_16k_st.shape[0]}, skipped."
            )
            continue

        saved_16k_names = [""] * sources_16k_st.shape[0]
        if args.save_enhanced or args.save_8k_sources:
            saved_16k_names = save_sources(
                args=args,
                enhanced_dir=enhanced_dir,
                fileid=fileid,
                mic_file=mic_path.name,
                sources_16k_st=sources_16k_st,
                sources_8k_st=sources_8k_st,
            )

        matches = match_sources_to_targets(sources_16k_st, refs, args.sample_rate)
        duration_sec = len(noisy_ref_for_metrics) / float(sr)
        for ref in refs:
            if ref.speaker_id not in matches:
                skipped_missing_match += 1
                print(f"fileid={fileid}, spk{ref.speaker_id}: no SepFormer source match, skipped.")
                continue
            source_idx, match_score = matches[ref.speaker_id]
            enhanced_file = saved_16k_names[source_idx] if source_idx < len(saved_16k_names) else ""
            scene_results.append(
                append_result_for_target(
                    args=args,
                    whisper_model=whisper_model,
                    fileid=fileid,
                    mic_file=mic_path.name,
                    duration_sec=duration_sec,
                    sr=sr,
                    ref=ref,
                    enhanced_for_asr=sources_16k_st[source_idx],
                    noisy_ref_for_metrics=noisy_ref_for_metrics,
                    source_idx=source_idx,
                    match_score=match_score,
                    enhanced_file=enhanced_file,
                    sepformer_sec=sepformer_sec,
                )
            )

    target_label = "allspk" if args.target_speaker_id <= 0 else f"spk{args.target_speaker_id}"
    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_sepformer_wer_details_{target_label}.csv"
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
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "enhancement": "speechbrain_sepformer_single_mic",
        "sepformer_source": args.sepformer_source,
        "sepformer_savedir": str(args.sepformer_savedir),
        "sample_rate": args.sample_rate,
        "sepformer_sample_rate": args.sepformer_sample_rate,
        "mic_channel": args.mic_channel,
        "num_sources": args.num_sources,
        "target_speaker_id": args.target_speaker_id,
        "source_matching": "oracle_si_sdr_clean_audio",
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "target_text_references": sum(len(v) for v in references_by_fileid.values()),
        "evaluated_utterances": len(scene_results),
        "skipped_no_gt_targets": skipped_no_gt_targets,
        "skipped_missing_match": skipped_missing_match,
        "truncated_target_ref_groups": truncated_target_refs,
        "overall": all_summary,
        "by_method": by_method,
    }

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_sepformer_wer_summary_{target_label}.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== SEPFORMER ASR SUMMARY =====")
    print(f"Target speaker: {'all' if args.target_speaker_id <= 0 else f'spk{args.target_speaker_id}'}")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Skipped scenes without target text refs: {summary['skipped_no_gt_targets']}")
    print(f"Skipped missing source matches: {summary['skipped_missing_match']}")
    for method, stats in by_method.items():
        print(
            f"{method}: utterances={stats['evaluated_utterances']}, "
            f"corpus WER={stats['corpus_wer']:.4f}, "
            f"mean sample WER={stats['mean_sample_wer']:.4f}, "
            f"mean total RTF={stats['mean_total_rtf']:.3f}"
        )
        print(
            f"{method}: mean SDRi={stats['mean_sdri']:.4f}, "
            f"mean SISDRi={stats['mean_sisdri']:.4f}, "
            f"mean PESQ={stats['mean_wb_pesq']:.4f}"
        )
    print(f"Saved details: {details_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
