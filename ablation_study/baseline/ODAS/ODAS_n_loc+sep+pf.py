#!/usr/bin/env python3
"""Batch offline ODAS -> predicted-DoA selection -> Whisper evaluation.

This mirrors the reporting of ``HARK_n_loc+sep+pf.py`` while keeping ODAS as
the localisation and separation system:

    four-channel mixture WAV
        -> ffmpeg PCM conversion
        -> odaslive with a scene-specific copy of ours.cfg
        -> 3 mono ODAS output slots
        -> predicted DoA per slot from tracks_*.json
        -> choose the predicted DoA nearest to spk1's *ground-truth* DoA
        -> Whisper small, WER, SDRi, and SI-SDRi

The DoA selection is ground-truth assisted and must be reported as such.  It
is directly comparable to the HARK IPD baseline's ``nearest_spk1_gt_doa``
selection rule; it is not a deployable source-selection policy.

Run this script from Linux/WSL, where ``odaslive`` and ``ffmpeg`` are on PATH.
All default paths are derived at runtime from this file, so they remain Linux
paths when this repository is mounted at /home/shaozi/ucl/code/audition_pipeline.
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
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import resample_poly
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
SCRIPT_STEM = Path(__file__).stem


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_seconds(device: str, fn: Callable[[], object]) -> Tuple[object, float]:
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
    dp = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(ref_words) + 1)
    dp[0, :] = np.arange(len(hyp_words) + 1)
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = int(ref_words[i - 1] != hyp_words[j - 1])
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[-1, -1])


def wer(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return (0.0 if not hyp_words else 1.0, len(hyp_words), 0)
    edits = edit_distance_words(ref_words, hyp_words)
    return edits / len(ref_words), edits, len(ref_words)


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


def parse_speaker_id(path_or_name: Path | str) -> int:
    match = re.search(r"spk(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse speaker id from: {path_or_name}")
    return int(match.group(1))


def circular_angle_error_deg(predicted: float, target: float) -> float:
    return float(abs((predicted - target + 180.0) % 360.0 - 180.0))


def circular_mean_deg(samples: Sequence[Tuple[float, float]]) -> float:
    """Weighted circular mean of ``(azimuth_deg, weight)`` samples."""
    if not samples:
        raise ValueError("Cannot calculate a circular mean without samples.")
    sin_sum = sum(math.sin(math.radians(angle)) * weight for angle, weight in samples)
    cos_sum = sum(math.cos(math.radians(angle)) * weight for angle, weight in samples)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


def load_multichannel_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if sample_rate != target_sr:
        divisor = math.gcd(sample_rate, target_sr)
        wav = np.stack(
            [
                resample_poly(wav[:, channel], target_sr // divisor, sample_rate // divisor).astype(np.float32)
                for channel in range(wav.shape[1])
            ],
            axis=1,
        )
        sample_rate = target_sr
    return wav, sample_rate


def load_mono_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sample_rate = sf.read(str(path), always_2d=True)
    mono = wav[:, 0].astype(np.float32)
    if sample_rate != target_sr:
        divisor = math.gcd(sample_rate, target_sr)
        mono = resample_poly(mono, target_sr // divisor, sample_rate // divisor).astype(np.float32)
        sample_rate = target_sr
    return mono, sample_rate


def si_sdr_db(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = prediction.astype(np.float64) - np.mean(prediction)
    target = target.astype(np.float64) - np.mean(target)
    scale = np.dot(prediction, target) / (np.dot(target, target) + eps)
    projection = scale * target
    noise = prediction - projection
    return float(10.0 * np.log10((np.sum(projection**2) + eps) / (np.sum(noise**2) + eps)))


def sdr_db(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    prediction = prediction.astype(np.float64)
    target = target.astype(np.float64)
    return float(10.0 * np.log10((np.sum(target**2) + eps) / (np.sum((target - prediction) ** 2) + eps)))


def compute_audio_quality_metrics(
    enhanced: np.ndarray,
    clean_path: Optional[Path],
    noisy_reference: np.ndarray,
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
    clean, clean_rate = load_mono_audio(clean_path, sample_rate)
    if clean_rate != sample_rate:
        return empty
    length = min(len(enhanced), len(clean), len(noisy_reference))
    if length == 0:
        return empty
    enhanced, clean, noisy_reference = enhanced[:length], clean[:length], noisy_reference[:length]
    input_sdr = sdr_db(noisy_reference, clean)
    output_sdr = sdr_db(enhanced, clean)
    input_si_sdr = si_sdr_db(noisy_reference, clean)
    output_si_sdr = si_sdr_db(enhanced, clean)
    return {
        "input_sdr": input_sdr,
        "sdr": output_sdr,
        "sdr_i": output_sdr - input_sdr,
        "input_si_sdr": input_si_sdr,
        "si_sdr": output_si_sdr,
        "si_sdr_i": output_si_sdr - input_si_sdr,
    }


def unique_mic_files(mic_dir: Path, max_items: int) -> List[Path]:
    paths = sorted(mic_dir.glob("mic_fileid_*_doa*_3spk.wav"), key=lambda p: (parse_fileid(p), parse_doa(p), p.name))
    return paths if max_items <= 0 else paths[:max_items]


def group_by_fileid(paths: Iterable[Path]) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in paths:
        grouped.setdefault(parse_fileid(path), []).append(path)
    return grouped


def representative_mic(paths: Sequence[Path]) -> Path:
    """The dataset stores the same mixture once per scene speaker; process it once."""
    return sorted(paths, key=lambda p: (parse_doa(p), p.name))[0]


@dataclass(frozen=True)
class TargetReference:
    fileid: int
    speaker_id: int
    gt_doa: int
    text_path: Path
    clean_path: Optional[Path]


@dataclass
class OdasRunRecord:
    fileid: int
    mic_file: str
    runtime_cfg: str
    output_dir: str
    command: str
    returncode: int
    elapsed_sec: float
    stdout_log: str
    stderr_log: str
    predicted_slots: str


@dataclass
class EvalRecord:
    fileid: int
    mic_file: str
    method: str
    audio_stage: str
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


def load_spk1_references(text_dir: Path, clean_dir: Path) -> Dict[int, TargetReference]:
    clean_by_key: Dict[Tuple[int, int], Path] = {}
    for path in clean_dir.glob("clean_fileid_*_doa*_spk1.wav"):
        clean_by_key[(parse_fileid(path), parse_doa(path))] = path

    refs: Dict[int, TargetReference] = {}
    for path in sorted(text_dir.glob("text_fileid_*_doa*_spk1.txt")):
        fileid, doa = parse_fileid(path), parse_doa(path)
        refs[fileid] = TargetReference(fileid, 1, doa, path, clean_by_key.get((fileid, doa)))
    return refs


def iter_odas_frames(path: Path) -> Iterator[dict]:
    """ODAS writes adjacent JSON objects, rather than a JSON array."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return
        frame, position = decoder.raw_decode(text, position)
        if isinstance(frame, dict):
            yield frame


def read_predicted_slot_doas(
    tracks_json: Path,
    sample_rate: int,
    hop_size: int,
    num_sources: int,
    warmup_seconds: float,
    min_activity: float,
) -> List[Tuple[int, int, float, float, int]]:
    """Return ``(slot, track_id, predicted_doa, mean_activity, observations)``.

    ODAS SSS slots are the positions in each ``src`` list.  Only ``tag=dynamic``
    entries are valid source tracks; the initial ID-0 / empty-tag records are
    intentionally ignored.  The DoA is an activity-weighted circular mean after
    startup, so it is a scene-level label for a slot rather than a frame-wise
    re-ordering of audio.
    """
    angle_samples: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    track_weights: Dict[int, Counter[int]] = defaultdict(Counter)
    activities: Dict[int, List[float]] = defaultdict(list)

    for frame in iter_odas_frames(tracks_json):
        timestamp = frame.get("timeStamp")
        if not isinstance(timestamp, (int, float)):
            continue
        seconds = float(timestamp) * hop_size / sample_rate
        if seconds < warmup_seconds:
            continue
        for slot, source in enumerate(frame.get("src", [])):
            if slot >= num_sources or source.get("tag") != "dynamic":
                continue
            try:
                activity = float(source.get("activity", 0.0))
                x, y = float(source["x"]), float(source["y"])
                track_id = int(source["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if activity < min_activity:
                continue
            azimuth = math.degrees(math.atan2(y, x)) % 360.0
            angle_samples[slot].append((azimuth, activity))
            track_weights[slot][track_id] += activity
            activities[slot].append(activity)

    results: List[Tuple[int, int, float, float, int]] = []
    for slot in range(num_sources):
        samples = angle_samples.get(slot, [])
        if not samples:
            continue
        dominant_track = track_weights[slot].most_common(1)[0][0]
        results.append(
            (
                slot,
                dominant_track,
                circular_mean_deg(samples),
                float(np.mean(activities[slot])),
                len(samples),
            )
        )
    return results


def scene_dir(args: argparse.Namespace, fileid: int) -> Path:
    return args.out_dir / "odas_outputs" / f"fileid_{fileid}"


def stage_raw_path(scene: Path, stage: str) -> Path:
    return scene / f"{stage}_fileid_{parse_fileid(scene.name)}.raw"


def stage_slot_wav(scene: Path, stage: str, slot: int, predicted_doa: float) -> Path:
    return scene / f"odas_{stage}_preddoa{round(predicted_doa) % 360:03d}_slot{slot}.wav"


def replace_path_in_cfg(text: str, section: str, new_path: Path) -> str:
    pattern = rf'(?s)(\b{re.escape(section)}\s*:\s*\{{.*?\bpath\s*=\s*")[^"]*(")'
    updated, count = re.subn(pattern, rf'\g<1>{new_path}\g<2>', text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace the path in config section '{section}'.")
    return updated


def make_runtime_cfg(args: argparse.Namespace, mic_raw: Path, scene: Path, fileid: int) -> Path:
    cfg = args.cfg_template.read_text(encoding="utf-8")
    paths = {
        "raw": mic_raw,
        "potential": scene / f"potential_fileid_{fileid}.json",
        "tracked": scene / f"tracks_fileid_{fileid}.json",
        "separated": scene / f"separated_fileid_{fileid}.raw",
        "postfiltered": scene / f"postfiltered_fileid_{fileid}.raw",
    }
    for section, output_path in paths.items():
        cfg = replace_path_in_cfg(cfg, section, output_path)
    runtime_cfg = scene / f"runtime_fileid_{fileid}.cfg"
    runtime_cfg.write_text(cfg, encoding="utf-8")
    return runtime_cfg


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: Sequence[str], stdout_log: Path, stderr_log: Path) -> Tuple[int, float]:
    start = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    stdout_log.write_text(process.stdout or "", encoding="utf-8")
    stderr_log.write_text(process.stderr or "", encoding="utf-8")
    return int(process.returncode), elapsed


def extract_slot_wavs(
    args: argparse.Namespace,
    scene: Path,
    stage: str,
    slot_doas: Sequence[Tuple[int, int, float, float, int]],
) -> List[Path]:
    raw_path = stage_raw_path(scene, stage)
    if not raw_path.is_file():
        raise FileNotFoundError(f"ODAS did not write expected {stage} RAW: {raw_path}")
    wavs: List[Path] = []
    for slot, _, predicted_doa, _, _ in slot_doas:
        output = stage_slot_wav(scene, stage, slot, predicted_doa)
        command = [
            args.ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(args.sample_rate),
            "-ac",
            str(args.num_sources),
            "-i",
            str(raw_path),
            "-af",
            f"pan=mono|c0=c{slot}",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg slot extraction failed:\n{process.stderr}")
        wavs.append(output)
    return wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ODAS + predicted-DoA spk1 selection + Whisper evaluation.")
    parser.add_argument("--mode", choices=("run_odas", "eval_odas", "both"), default="both")
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--cfg_template", type=Path, default=OFFLINE_ROOT / "ours.cfg")
    parser.add_argument("--odaslive", type=str, default="odaslive")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--audio_stage", choices=("separated", "postfiltered"), default="postfiltered")
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--hop_size", type=int, default=128)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--warmup_seconds", type=float, default=1.0)
    parser.add_argument("--min_activity", type=float, default=0.05)
    parser.add_argument("--max_items", type=int, default=0, help="Number of mic entries to inspect; 0 means all scenes.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip an ODAS scene if its selected-stage WAVs and track JSON exist.")
    return parser.parse_args()


def check_executable(command: str, label: str) -> None:
    if shutil.which(command) is None and not Path(command).is_file():
        raise FileNotFoundError(f"{label} was not found: {command}. Ensure it is on PATH or pass its full Linux path.")


def run_odas(args: argparse.Namespace) -> List[OdasRunRecord]:
    check_executable(args.odaslive, "odaslive")
    check_executable(args.ffmpeg, "ffmpeg")
    grouped = group_by_fileid(unique_mic_files(args.mic_dir, args.max_items))
    logs_dir = args.out_dir / "odas_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records: List[OdasRunRecord] = []

    for fileid, scene_mics in tqdm(grouped.items(), desc="ODAS", unit="scene"):
        mic_path = representative_mic(scene_mics)
        scene = scene_dir(args, fileid)
        scene.mkdir(parents=True, exist_ok=True)
        tracks_json = scene / f"tracks_fileid_{fileid}.json"
        existing_slots = list(scene.glob(f"odas_{args.audio_stage}_preddoa*_slot*.wav"))
        if args.skip_existing and tracks_json.is_file() and existing_slots:
            slots = read_predicted_slot_doas(
                tracks_json, args.sample_rate, args.hop_size, args.num_sources, args.warmup_seconds, args.min_activity
            )
            records.append(
                OdasRunRecord(fileid, str(mic_path), "", str(scene), "SKIPPED_EXISTING", 0, 0.0, "", "", format_slots(slots))
            )
            continue

        mic_raw = scene / f"mic_fileid_{fileid}.raw"
        conversion = [
            args.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mic_path),
            "-f", "s16le", "-ar", str(args.sample_rate), "-ac", "4", str(mic_raw),
        ]
        conversion_code, _ = run_command(
            conversion, logs_dir / f"odas_fileid_{fileid}.ffmpeg.stdout.log", logs_dir / f"odas_fileid_{fileid}.ffmpeg.stderr.log"
        )
        if conversion_code != 0:
            records.append(OdasRunRecord(fileid, str(mic_path), "", str(scene), command_text(conversion), conversion_code, 0.0, "", "", ""))
            continue

        runtime_cfg = make_runtime_cfg(args, mic_raw, scene, fileid)
        odas_command = [args.odaslive, "-c", str(runtime_cfg)]
        stdout_log = logs_dir / f"odas_fileid_{fileid}.stdout.log"
        stderr_log = logs_dir / f"odas_fileid_{fileid}.stderr.log"
        returncode, elapsed = run_command(odas_command, stdout_log, stderr_log)
        slots: List[Tuple[int, int, float, float, int]] = []
        if returncode == 0:
            try:
                slots = read_predicted_slot_doas(
                    tracks_json, args.sample_rate, args.hop_size, args.num_sources, args.warmup_seconds, args.min_activity
                )
                if not slots:
                    raise ValueError("No dynamic ODAS tracks passed the activity threshold.")
                extract_slot_wavs(args, scene, args.audio_stage, slots)
                (scene / "slot_doa_manifest.json").write_text(
                    json.dumps(
                        [
                            {"slot": slot, "track_id": track_id, "predicted_doa": doa, "mean_activity": activity, "observations": count}
                            for slot, track_id, doa, activity, count in slots
                        ],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                returncode = 1
                with stderr_log.open("a", encoding="utf-8") as log:
                    log.write(f"\nPost-processing failed: {exc}\n")
        records.append(
            OdasRunRecord(
                fileid, str(mic_path), str(runtime_cfg), str(scene), command_text(odas_command), returncode, elapsed,
                str(stdout_log), str(stderr_log), format_slots(slots),
            )
        )

    manifest = args.out_dir / "odas_run_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OdasRunRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    return records


def format_slots(slots: Sequence[Tuple[int, int, float, float, int]]) -> str:
    return ";".join(f"slot{slot}:id{track_id}:doa{doa:.1f}" for slot, track_id, doa, _, _ in slots)


def load_slot_manifest(scene: Path) -> List[Tuple[int, int, float, float, int]]:
    manifest = scene / "slot_doa_manifest.json"
    if not manifest.is_file():
        return []
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        (int(row["slot"]), int(row["track_id"]), float(row["predicted_doa"]), float(row["mean_activity"]), int(row["observations"]))
        for row in rows
    ]


def evaluate_odas(args: argparse.Namespace) -> List[EvalRecord]:
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")
    refs = load_spk1_references(args.text_dir, args.clean_dir)
    grouped = group_by_fileid(unique_mic_files(args.mic_dir, args.max_items))
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)
    records: List[EvalRecord] = []
    skipped = Counter()

    for fileid, scene_mics in tqdm(grouped.items(), desc="Eval-ODAS", unit="scene"):
        ref = refs.get(fileid)
        if ref is None:
            skipped["no_spk1_reference"] += 1
            continue
        scene = scene_dir(args, fileid)
        slots = load_slot_manifest(scene)
        if not slots:
            tracks_json = scene / f"tracks_fileid_{fileid}.json"
            if tracks_json.is_file():
                slots = read_predicted_slot_doas(
                    tracks_json, args.sample_rate, args.hop_size, args.num_sources, args.warmup_seconds, args.min_activity
                )
        candidates = [
            (stage_slot_wav(scene, args.audio_stage, slot, doa), slot, doa)
            for slot, _, doa, _, _ in slots
        ]
        candidates = [candidate for candidate in candidates if candidate[0].is_file()]
        if not candidates:
            skipped["no_odas_candidates"] += 1
            continue
        selected_wav, selected_slot, predicted_doa = min(
            candidates, key=lambda item: (circular_angle_error_deg(item[2], ref.gt_doa), item[1])
        )
        enhanced, sample_rate = load_mono_audio(selected_wav, args.sample_rate)
        reference = ref.text_path.read_text(encoding="utf-8").strip()

        def run_asr() -> object:
            return whisper_model.transcribe(enhanced, language=args.language, fp16=args.whisper_device.startswith("cuda"))

        asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
        hypothesis = asr_out.get("text", "").strip()  # type: ignore[union-attr]
        sample_wer, edits, ref_words = wer(reference, hypothesis)
        mic_path = representative_mic(scene_mics)
        noisy, _ = load_multichannel_audio(mic_path, sample_rate)
        noisy_ref = noisy[:, 0]
        metrics = compute_audio_quality_metrics(enhanced, ref.clean_path, noisy_ref, sample_rate)
        duration = len(noisy_ref) / sample_rate
        records.append(
            EvalRecord(
                fileid=fileid,
                mic_file=str(mic_path),
                method="ODAS-SSL-SST-DGSS-MS",
                audio_stage=args.audio_stage,
                speaker_id=1,
                duration_sec=duration,
                gt_doa=ref.gt_doa,
                selected_wav=str(selected_wav),
                selection_strategy="nearest_predicted_doa_to_spk1_gt_doa",
                selected_source_index=selected_slot,
                predicted_doa=predicted_doa,
                doa_error_deg=circular_angle_error_deg(predicted_doa, ref.gt_doa),
                separated_wav_count=len(candidates),
                gt_text_file=str(ref.text_path),
                wer=sample_wer,
                edit_distance=edits,
                ref_words=ref_words,
                input_sdr=metrics["input_sdr"],
                sdr=metrics["sdr"],
                sdr_i=metrics["sdr_i"],
                input_si_sdr=metrics["input_si_sdr"],
                si_sdr=metrics["si_sdr"],
                si_sdr_i=metrics["si_sdr_i"],
                reference=reference,
                hypothesis=hypothesis,
                whisper_sec=whisper_sec,
                total_sec=whisper_sec,
                whisper_rtf=whisper_sec / duration,
                total_rtf=whisper_sec / duration,
                under_realtime=int(whisper_sec < duration),
            )
        )
        print(
            f"fileid={fileid} spk1: GT DoA={ref.gt_doa}°, "
            f"selected slot={selected_slot}, predicted DoA={predicted_doa:.1f}°, "
            f"DoA error={circular_angle_error_deg(predicted_doa, ref.gt_doa):.1f}°, "
            f"WER={sample_wer:.4f}"
        )

    details = args.out_dir / f"pipeline_whisper_{args.whisper_model}_odas_wer_details_spk1.csv"
    with details.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvalRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in records)
    summary = summarize(records)
    summary.update(
        {
            "mic_dir": str(args.mic_dir),
            "clean_dir": str(args.clean_dir),
            "text_dir": str(args.text_dir),
            "cfg_template": str(args.cfg_template),
            "audio_stage": args.audio_stage,
            "selection_strategy": "nearest_predicted_doa_to_spk1_gt_doa",
            "target_speaker_id": 1,
            "whisper_model": args.whisper_model,
            "whisper_device": args.whisper_device,
            "warmup_seconds": args.warmup_seconds,
            "min_activity": args.min_activity,
            "skipped": dict(skipped),
        }
    )
    summary_path = args.out_dir / f"pipeline_whisper_{args.whisper_model}_odas_wer_summary_spk1.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== ODAS ASR SUMMARY (spk1; GT-DoA-assisted selection) =====")
    print(f"Evaluated utterances: {summary['evaluated_utterances']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Mean sample WER: {summary['mean_sample_wer']:.4f}")
    print(f"Mean SDRi: {summary['mean_sdri']:.4f}")
    print(f"Mean SI-SDRi: {summary['mean_sisdri']:.4f}")
    print(f"Saved details: {details}")
    print(f"Saved summary: {summary_path}")
    return records


def summarize(rows: Sequence[EvalRecord]) -> Dict[str, float | int]:
    def mean_optional(field: str) -> float:
        values = [float(value) for row in rows for value in [getattr(row, field)] if value is not None and np.isfinite(value)]
        return float(np.mean(values)) if values else 0.0

    edits = sum(int(row.edit_distance or 0) for row in rows)
    words = sum(int(row.ref_words or 0) for row in rows)
    wers = [float(row.wer) for row in rows if row.wer is not None]
    return {
        "evaluated_utterances": len(rows),
        "corpus_wer": edits / words if words else 0.0,
        "mean_sample_wer": float(np.mean(wers)) if wers else 0.0,
        "mean_input_sdr": mean_optional("input_sdr"),
        "mean_sdr": mean_optional("sdr"),
        "mean_sdri": mean_optional("sdr_i"),
        "mean_input_si_sdr": mean_optional("input_si_sdr"),
        "mean_si_sdr": mean_optional("si_sdr"),
        "mean_sisdri": mean_optional("si_sdr_i"),
        "mean_whisper_sec": mean_optional("whisper_sec"),
        "mean_total_rtf": mean_optional("total_rtf"),
        "under_realtime_count": sum(row.under_realtime for row in rows),
        "under_realtime_rate": float(np.mean([row.under_realtime for row in rows])) if rows else 0.0,
    }


def main() -> None:
    args = parse_args()
    for directory, label in ((args.mic_dir, "Mic"), (args.clean_dir, "Clean"), (args.text_dir, "Text")):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} directory not found: {directory}")
    if not args.cfg_template.is_file():
        raise FileNotFoundError(f"ODAS config template not found: {args.cfg_template}")
    if args.num_sources != 3:
        raise ValueError("This runner currently expects the three-slot ODAS configuration (num_sources=3).")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("run_odas", "both"):
        run_records = run_odas(args)
        failed = [record for record in run_records if record.returncode != 0]
        if failed:
            raise RuntimeError(f"ODAS failed for {len(failed)} scene(s); inspect {args.out_dir / 'odas_logs'}.")
    if args.mode in ("eval_odas", "both"):
        evaluate_odas(args)


if __name__ == "__main__":
    main()
