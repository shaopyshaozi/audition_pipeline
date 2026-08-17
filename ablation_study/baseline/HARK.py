"""
Official PyHARK baseline wrapper and evaluator.

This script does not reimplement MUSIC, SourceTracker, or GHDSS numerically.
It calls official PyHARK nodes:

    4-channel wav
        -> PyHARK:
           LocalizeMUSIC -> SourceTracker -> GHDSS
        -> separated wavs
        -> Whisper small
        -> WER / SDRi / SI-SDRi

Usage is intentionally two-stage:

1. run_pyhark:
   call the PyHARK runner once per test scene.
2. eval_official:
   evaluate the separated wavs produced by PyHARK with the same Whisper
   and metric code used by the other baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
import subprocess
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
DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"
SCRIPT_STEM = Path(__file__).stem


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
    parser = argparse.ArgumentParser(description="Call official PyHARK and evaluate its separated wavs.")
    parser.add_argument(
        "--mode",
        choices=("run_pyhark", "run_official", "eval_official", "both"),
        default="both",
        help="Run PyHARK, evaluate existing PyHARK/HARK outputs, or do both. run_official is kept as an alias.",
    )
    parser.add_argument("--mic_dir", type=Path, default=DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--official_output_dir", type=Path, default=None)
    parser.add_argument("--pyhark_script", type=Path, default=OFFLINE_ROOT / "pyhark_localizemusic_ghdss.py")
    parser.add_argument("--localization_tf", type=Path, default=None, help="PyHARK LocalizeMUSIC transfer function zip.")
    parser.add_argument("--separation_tf", type=Path, default=None, help="PyHARK GHDSS transfer function zip. Defaults to --localization_tf.")
    parser.add_argument("--channel_count", type=int, default=4)
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--frame_shift", type=int, default=160)
    parser.add_argument("--music_min_freq", type=int, default=500)
    parser.add_argument("--music_max_freq", type=int, default=2800)
    parser.add_argument("--music_window", type=int, default=50)
    parser.add_argument("--music_period", type=int, default=50)
    parser.add_argument("--tracker_thresh", type=float, default=25.0)
    parser.add_argument("--tracker_pause_length", type=float, default=1200.0)
    parser.add_argument("--tracker_min_src_interval", type=float, default=20.0)
    parser.add_argument("--output_bits", type=str, default="int16")
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
        choices=("oracle_sisdr", "source_index", "doa_csv"),
        default="oracle_sisdr",
        help=(
            "How to choose the target separated stream. oracle_sisdr is an upper-bound "
            "selection for source-order ambiguity; doa_csv is preferred if your HARK "
            "network logs source DOAs."
        ),
    )
    parser.add_argument("--source_index", type=int, default=0)
    parser.add_argument(
        "--doa_csv",
        type=Path,
        default=None,
        help=(
            "CSV with HARK output DOAs. Supported columns include fileid, doa_deg, and "
            "either source_index or wav_path/wav_file."
        ),
    )
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument(
        "--target_speaker_id",
        type=int,
        default=1,
        help="Speaker id to evaluate. Use 0 to evaluate all target references.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    return parser.parse_args()


def scene_output_dir(args: argparse.Namespace, fileid: int) -> Path:
    root = args.official_output_dir or (args.out_dir / "official_hark_outputs")
    return root / f"fileid_{fileid}"


def scene_output_prefix(args: argparse.Namespace, fileid: int, mic_path: Path) -> Path:
    return scene_output_dir(args, fileid) / "sep_{srcid}_"


def find_scene_wavs(args: argparse.Namespace, fileid: int) -> List[Path]:
    out_dir = scene_output_dir(args, fileid)
    if not out_dir.exists():
        return []
    return sorted(path for path in out_dir.glob(args.official_output_glob) if path.is_file())


def render_pyhark_command(args: argparse.Namespace, mic_path: Path, fileid: int) -> List[str]:
    if args.localization_tf is None:
        raise FileNotFoundError("--localization_tf is required in run_pyhark/both mode.")
    if not args.localization_tf.is_file():
        raise FileNotFoundError(f"Localization transfer function not found: {args.localization_tf}")
    if args.separation_tf is not None and not args.separation_tf.is_file():
        raise FileNotFoundError(f"Separation transfer function not found: {args.separation_tf}")
    if not args.pyhark_script.is_file():
        raise FileNotFoundError(f"PyHARK runner script not found: {args.pyhark_script}")

    out_dir = scene_output_dir(args, fileid)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(args.pyhark_script),
        "--input_wav",
        str(mic_path),
        "--output_prefix",
        str(scene_output_prefix(args, fileid, mic_path)),
        "--localization_tf",
        str(args.localization_tf),
        "--channel_count",
        str(args.channel_count),
        "--sample_rate",
        str(args.sample_rate),
        "--num_sources",
        str(args.num_sources),
        "--frame_length",
        str(args.frame_length),
        "--frame_shift",
        str(args.frame_shift),
        "--music_min_freq",
        str(args.music_min_freq),
        "--music_max_freq",
        str(args.music_max_freq),
        "--music_window",
        str(args.music_window),
        "--music_period",
        str(args.music_period),
        "--tracker_thresh",
        str(args.tracker_thresh),
        "--tracker_pause_length",
        str(args.tracker_pause_length),
        "--tracker_min_src_interval",
        str(args.tracker_min_src_interval),
        "--output_bits",
        str(args.output_bits),
    ]
    if args.separation_tf is not None:
        command.extend(["--separation_tf", str(args.separation_tf)])
    return command


def run_official_hark(args: argparse.Namespace) -> List[HarkRunRecord]:
    import_check = subprocess.run([sys.executable, "-c", "import hark"], capture_output=True, text=True, check=False)
    if import_check.returncode != 0:
        raise RuntimeError(
            "PyHARK is not importable in this Python environment.\n"
            "Verify with:\n"
            f"  {sys.executable} -c 'import hark; print(hark)'\n"
            f"stderr:\n{import_check.stderr}"
        )

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    records: List[HarkRunRecord] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "official_hark_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for fileid, target_paths in tqdm(grouped.items(), desc="Official-HARK", unit="scene"):
        mic_path = choose_representative_mic(target_paths)
        out_dir = scene_output_dir(args, fileid)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = find_scene_wavs(args, fileid)
        if args.skip_existing and existing:
            records.append(
                HarkRunRecord(
                    fileid=fileid,
                    mic_file=str(mic_path),
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

        command = render_pyhark_command(args, mic_path, fileid)
        stdout_log = logs_dir / f"pyhark_fileid_{fileid}.stdout.log"
        stderr_log = logs_dir / f"pyhark_fileid_{fileid}.stderr.log"
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        wavs = find_scene_wavs(args, fileid)
        records.append(
            HarkRunRecord(
                fileid=fileid,
                mic_file=str(mic_path),
                output_dir=str(out_dir),
                output_prefix=str(scene_output_prefix(args, fileid, mic_path)),
                command=" ".join(command),
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


def load_doa_csv(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wav_matches_doa_row(wav_path: Path, row: Dict[str, str], sorted_wavs: Sequence[Path]) -> bool:
    if row.get("wav_path") and Path(row["wav_path"]).name == wav_path.name:
        return True
    if row.get("wav_file") and Path(row["wav_file"]).name == wav_path.name:
        return True
    if row.get("source_index") not in (None, ""):
        try:
            return sorted_wavs[int(row["source_index"])] == wav_path
        except (ValueError, IndexError):
            return False
    return False


def get_doa_for_wav(
    doa_rows: Sequence[Dict[str, str]],
    fileid: int,
    wav_path: Path,
    sorted_wavs: Sequence[Path],
) -> Optional[float]:
    for row in doa_rows:
        if int(float(row.get("fileid", "-1"))) != fileid:
            continue
        if not wav_matches_doa_row(wav_path, row, sorted_wavs):
            continue
        value = row.get("doa_deg") or row.get("azimuth") or row.get("azimuth_deg")
        if value is None or value == "":
            return None
        return float(value)
    return None


def choose_official_output(
    *,
    args: argparse.Namespace,
    wavs: Sequence[Path],
    ref: TargetReference,
    noisy_ref: np.ndarray,
    sample_rate: int,
    doa_rows: Sequence[Dict[str, str]],
) -> Tuple[Path, Optional[int], Optional[float], Optional[float]]:
    if not wavs:
        raise ValueError("No official HARK separated wavs found for this scene.")

    if args.source_selection == "source_index":
        if args.source_index < 0 or args.source_index >= len(wavs):
            raise ValueError(f"--source_index {args.source_index} out of range for {len(wavs)} wavs.")
        return wavs[args.source_index], args.source_index, None, None

    if args.source_selection == "doa_csv":
        if not doa_rows:
            raise ValueError("--source_selection doa_csv requires --doa_csv.")
        candidates = []
        for idx, wav_path in enumerate(wavs):
            doa = get_doa_for_wav(doa_rows, ref.fileid, wav_path, wavs)
            if doa is None:
                continue
            candidates.append((circular_angle_error_deg(doa, ref.gt_doa), idx, wav_path, doa))
        if not candidates:
            raise ValueError(f"No matching DOA rows for fileid={ref.fileid}.")
        error, idx, wav_path, doa = min(candidates, key=lambda item: item[0])
        return wav_path, idx, doa, error

    best: Optional[Tuple[float, int, Path]] = None
    if ref.clean_path is None:
        raise ValueError("oracle_sisdr selection requires clean target wavs.")
    clean, _ = load_mono_audio(ref.clean_path, target_sr=sample_rate)
    for idx, wav_path in enumerate(wavs):
        enhanced, _ = load_mono_audio(wav_path, target_sr=sample_rate)
        min_len = min(len(enhanced), len(clean), len(noisy_ref))
        score = si_sdr_db(enhanced[:min_len], clean[:min_len]) if min_len > 0 else -np.inf
        if best is None or score > best[0]:
            best = (score, idx, wav_path)
    assert best is not None
    return best[2], best[1], None, None


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
    doa_rows = load_doa_csv(args.doa_csv)
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

        for ref in target_refs:
            try:
                selected_wav, selected_index, pred_doa, doa_error = choose_official_output(
                    args=args,
                    wavs=separated_wavs,
                    ref=ref,
                    noisy_ref=noisy_ref,
                    sample_rate=sr,
                    doa_rows=doa_rows,
                )
            except Exception as exc:
                skipped["selection_failed"] += 1
                print(f"fileid={fileid}, spk{ref.speaker_id}: output selection failed: {exc}")
                continue

            enhanced, _ = load_mono_audio(selected_wav, target_sr=sr)

            def run_asr():
                return whisper_model.transcribe(
                    enhanced,
                    language=args.language,
                    fp16=args.whisper_device.startswith("cuda"),
                )

            asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
            hyp_text = asr_out.get("text", "").strip()
            ref_text = ref.text_path.read_text(encoding="utf-8").strip()
            sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)
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
                    method="Official-HARK-LocalizeMUSIC-SourceTracker-GHDSS",
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
            "pyhark_runtime": True,
            "skipped": skipped,
        }
    )
    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_official_hark_wer_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== OFFICIAL PYHARK ASR SUMMARY =====")
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

    if args.mode in ("run_pyhark", "run_official", "both"):
        run_records = run_official_hark(args)
        failed = [row for row in run_records if row.returncode != 0]
        if failed:
            print(f"PyHARK failed on {len(failed)} scenes. Check official_hark_logs.")
            if args.mode == "both":
                raise RuntimeError("Stopping before evaluation because at least one HARK run failed.")

    if args.mode in ("eval_official", "both"):
        evaluate_official_outputs(args)


if __name__ == "__main__":
    main()
