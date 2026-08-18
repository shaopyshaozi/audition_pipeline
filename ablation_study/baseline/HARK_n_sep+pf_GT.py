"""
Official HARK baseline wrapper and evaluator.

This script does not implement localization or GHDSS in Python. It is for the
official HARK GT-DOA separation baseline:

    4-channel wav
        -> official HARK network:
           ConstantLocalization with spk-ordered ground-truth DOAs -> GHDSS
        -> spk-ordered separated wavs
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
Evaluation assumes HARK outputs follow the ConstantLocalization source order:
spk1 -> output index 0, spk2 -> output index 1, spk3 -> output index 2.
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
import xml.etree.ElementTree as ET
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
    gt_doas: str
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
        help="Official HARK .n network file. This GT-DOA script expects a ConstantLocalization node.",
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
        choices=("spk_order",),
        default="spk_order",
        help=(
            "Choose the enhanced stream by ConstantLocalization source order: "
            "spk1 -> output 0, spk2 -> output 1, spk3 -> output 2."
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
        choices=(1,),
        help="Speaker id to evaluate. This GT separation script compares spk1 only.",
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


def enhanced_output_path(args: argparse.Namespace, ref: TargetReference) -> Path:
    return scene_output_dir(args, ref.fileid) / (
        f"enhanced_fileid_{ref.fileid}_doa{ref.gt_doa}_spk{ref.speaker_id}.wav"
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
        ]
    )


def relabel_hark_outputs_in_place(
    args: argparse.Namespace,
    refs: Sequence[TargetReference],
) -> None:
    raw_wavs = find_raw_hark_wavs(args, refs[0].fileid if refs else -1)
    for ref in refs:
        source_index = ref.speaker_id - 1
        if source_index < 0 or source_index >= len(raw_wavs):
            continue
        source_path = raw_wavs[source_index]
        target_path = enhanced_output_path(args, ref)
        if source_path.resolve() == target_path.resolve():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            target_path.unlink()
        source_path.replace(target_path)


def all_enhanced_outputs_exist(args: argparse.Namespace, refs: Sequence[TargetReference]) -> bool:
    return bool(refs) and all(enhanced_output_path(args, ref).is_file() for ref in refs)


def materialize_spk_ordered_enhanced(
    args: argparse.Namespace,
    refs: Sequence[TargetReference],
    separated_wavs: Sequence[Path],
    sample_rate: int,
) -> Dict[Tuple[int, int], Tuple[Path, int, np.ndarray]]:
    outputs: Dict[Tuple[int, int], Tuple[Path, int, np.ndarray]] = {}
    raw_wavs = find_raw_hark_wavs(args, refs[0].fileid if refs else -1)
    for ref in refs:
        source_index = ref.speaker_id - 1
        output_path = enhanced_output_path(args, ref)
        if output_path.exists():
            audio, _ = load_mono_audio(output_path, target_sr=sample_rate)
            outputs[(ref.speaker_id, ref.gt_doa)] = (output_path, source_index, audio)
            continue
        if source_index < 0 or source_index >= len(raw_wavs):
            continue
        source_path = raw_wavs[source_index]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        source_path.replace(output_path)
        audio, _ = load_mono_audio(output_path, target_sr=sample_rate)
        outputs[(ref.speaker_id, ref.gt_doa)] = (output_path, source_index, audio)
    return outputs


def vector_float(values: Sequence[float | int]) -> str:
    body = " ".join(str(int(value)) if float(value).is_integer() else str(float(value)) for value in values)
    return f"<Vector<float> {body}>"


def scene_refs_in_speaker_order(
    refs_by_fileid: Dict[int, List[TargetReference]],
    fileid: int,
    target_paths: Sequence[Path],
) -> List[TargetReference]:
    refs = list(refs_by_fileid.get(fileid, []))
    refs.sort(key=lambda ref: (ref.speaker_id, ref.gt_doa, ref.text_path.name))
    if refs:
        return refs

    # Fallback for HARK-only runs without text refs. This preserves deterministic
    # behavior, but full spk-order evaluation requires text/clean references.
    return [
        TargetReference(
            fileid=fileid,
            speaker_id=idx + 1,
            gt_doa=parse_doa(path),
            text_path=Path(""),
            clean_path=None,
        )
        for idx, path in enumerate(sorted(target_paths, key=lambda p: (parse_doa(p), p.name)))
    ]


def scene_gt_doas_in_speaker_order(
    refs_by_fileid: Dict[int, List[TargetReference]],
    fileid: int,
    target_paths: Sequence[Path],
) -> List[int]:
    return [int(ref.gt_doa) for ref in scene_refs_in_speaker_order(refs_by_fileid, fileid, target_paths)]


def patch_hark_network_for_scene(
    args: argparse.Namespace,
    mic_path: Path,
    fileid: int,
    gt_doas: Sequence[int],
) -> Path:
    if args.hark_network is None:
        raise ValueError("--hark_network is required for --mode run_official or --mode both.")
    if not gt_doas:
        raise ValueError(f"No ground-truth DOAs available for fileid={fileid}.")

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
                parameter.set("value", vector_float(gt_doas))
            elif node_type == "ConstantLocalization" and name == "ELEVATIONS":
                parameter.set("value", vector_float([0] * len(gt_doas)))
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
    gt_doas: Sequence[int],
) -> List[str]:
    if args.hark_network is None:
        raise ValueError("--hark_network is required for --mode run_official or --mode both.")

    out_dir = scene_output_dir(args, fileid)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = scene_output_prefix(args, fileid, mic_path)
    runtime_network = patch_hark_network_for_scene(args, mic_path, fileid, gt_doas)
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
    scene_refs_by_fileid, _ = load_scene_references(
        args.text_dir,
        args.clean_dir,
        target_speaker_id=0,
    )
    records: List[HarkRunRecord] = []

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.out_dir / "official_hark_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for fileid, target_paths in tqdm(grouped.items(), desc="Official-HARK", unit="scene"):
        mic_path = choose_representative_mic(target_paths)
        scene_refs = scene_refs_in_speaker_order(scene_refs_by_fileid, fileid, target_paths)
        gt_doas = [int(ref.gt_doa) for ref in scene_refs]
        out_dir = scene_output_dir(args, fileid)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = find_scene_wavs(args, fileid)
        if args.skip_existing and existing and all_enhanced_outputs_exist(args, scene_refs):
            records.append(
                HarkRunRecord(
                    fileid=fileid,
                    mic_file=str(mic_path),
                    gt_doas=",".join(str(doa) for doa in gt_doas),
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
        command = render_hark_command(args, mic_path, fileid, gt_doas)
        stdout_log = logs_dir / f"hark_fileid_{fileid}.stdout.log"
        stderr_log = logs_dir / f"hark_fileid_{fileid}.stderr.log"
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        if proc.returncode == 0:
            relabel_hark_outputs_in_place(args, scene_refs)
        wavs = find_scene_wavs(args, fileid)
        records.append(
            HarkRunRecord(
                fileid=fileid,
                mic_file=str(mic_path),
                gt_doas=",".join(str(doa) for doa in gt_doas),
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
    scene_refs_by_fileid, target_refs_by_fileid = load_scene_references(
        args.text_dir,
        args.clean_dir,
        target_speaker_id=args.target_speaker_id,
    )
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

        enhanced_by_ref = materialize_spk_ordered_enhanced(
            args=args,
            refs=scene_refs_by_fileid.get(fileid, []),
            separated_wavs=separated_wavs,
            sample_rate=sr,
        )

        for ref in target_refs:
            ref_text = ref.text_path.read_text(encoding="utf-8").strip()
            try:
                selected = enhanced_by_ref.get((ref.speaker_id, ref.gt_doa))
                if selected is None:
                    raise ValueError(
                        f"No spk-ordered HARK output for spk{ref.speaker_id}, "
                        f"doa{ref.gt_doa}; got {len(separated_wavs)} separated wavs."
                    )
                selected_wav, selected_index, enhanced = selected

                def run_asr():
                    return whisper_model.transcribe(
                        enhanced,
                        language=args.language,
                        fp16=args.whisper_device.startswith("cuda"),
                    )

                asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
                hyp_text = asr_out.get("text", "").strip()
                sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)
                pred_doa = None
                doa_error = None
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
                    method="Official-HARK-GTDOA-ConstantLocalization-GHDSS",
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
