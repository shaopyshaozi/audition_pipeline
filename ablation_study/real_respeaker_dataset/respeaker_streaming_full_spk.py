"""
Official HARK loc+sep evaluation on the real ReSpeaker recordings.

This script replaces the original IPDNet -> DSENet -> streaming-ASR pipeline
with official HARK:

    long ReSpeaker recording
        -> full recording as one HARK input
        -> HARK loc+sep .n network via batchflow
        -> separated candidate wavs for the full scene
        -> SimulStreaming Whisper
        -> spk1 dominant WER / no-insertion WER
        -> spk2 non-dominant WER / no-insertion WER

Because the HARK loc+sep outputs do not reliably expose a stable speaker label
in the wav filename, the default selection is scene-level oracle WER: every
HARK candidate stream is transcribed, then the best stream is
selected separately for spk1 and spk2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import shutil
import socket
import string
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ABLATION_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ABLATION_ROOT.parent
BASELINE_ROOT = ABLATION_ROOT / "baseline"
SIMULSTREAMING_ROOT = PROJECT_ROOT / "Models" / "SimulStreaming"
REAL_RESPEAKER_ROOT = PROJECT_ROOT / "online" / "IPDNET" / "eval" / "Respeaker_real"
SCRIPT_STEM = Path(__file__).stem
STREAM_SAMPLE_RATE = 16000
STREAM_BYTES_PER_SAMPLE = 2
STREAM_BYTES_PER_SECOND = STREAM_SAMPLE_RATE * STREAM_BYTES_PER_SAMPLE


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
    return re.sub(r"\s+", " ", text).strip()


def word_error_stats(ref: str, hyp: str) -> Tuple[float, float, int, int, int, int, int]:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        insertions = m
        return (0.0 if m == 0 else 1.0, 0.0, insertions, 0, 0, insertions, 0)

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

    substitutions = 0
    deletions = 0
    insertions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                substitutions += int(cost)
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    edits = substitutions + deletions + insertions
    return edits / n, (substitutions + deletions) / n, edits, substitutions, deletions, insertions, n


def parse_respeaker_recording_name(path_or_name: Path | str) -> Tuple[int, int, Tuple[Optional[int], ...]]:
    name = Path(path_or_name).name
    match = re.match(r"fileid_(\d+)_sources_(\d+)_([^_]+)_([^_]+)_([^_]+)\.wav$", name)
    if not match:
        raise ValueError(f"Could not parse ReSpeaker recording name: {path_or_name}")
    fileid = int(match.group(1))
    source_count = int(match.group(2))
    doas: List[Optional[int]] = []
    for raw in match.groups()[2:]:
        doas.append(None if raw.upper() == "NA" else int(raw) % 360)
    return fileid, source_count, tuple(doas)


def resample_multichannel_audio(wav: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    wav = wav.astype(np.float32)
    if sr == target_sr:
        return wav, sr
    gcd = math.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    wav = np.stack(
        [resample_poly(wav[:, ch], up, down).astype(np.float32) for ch in range(wav.shape[1])],
        axis=1,
    )
    return wav, target_sr


def load_mono_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav[:, 0].astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return wav, sr


def mono_audio_to_pcm16(audio: np.ndarray, sr: int, target_sr: int = STREAM_SAMPLE_RATE) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def _decode_transcript_line(line: str) -> Dict[str, Any]:
    line = line.strip().strip("\0")
    if not line:
        return {}
    if line.startswith("{"):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {"raw": line}
    parts = line.split(maxsplit=2)
    if len(parts) == 3 and parts[0].replace(".", "", 1).isdigit() and parts[1].replace(".", "", 1).isdigit():
        return {
            "start": float(parts[0]) / 1000.0,
            "end": float(parts[1]) / 1000.0,
            "text": parts[2],
        }
    return {"raw": line}


def transcript_text(segments: Sequence[Dict[str, Any]]) -> str:
    return " ".join(
        str(item.get("text") or item.get("raw") or "").strip()
        for item in segments
        if str(item.get("text") or item.get("raw") or "").strip()
    )


class StreamingWhisperClient:
    def __init__(
        self,
        host: str,
        port: int,
        packet_ms: int,
        realtime: bool,
        connect_timeout: float,
    ):
        self.host = host
        self.port = port
        self.packet_ms = packet_ms
        self.realtime = realtime
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._transcripts: List[Dict[str, Any]] = []

    def connect(self) -> None:
        deadline = time.perf_counter() + self.connect_timeout
        last_error: Optional[BaseException] = None
        while time.perf_counter() < deadline:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=2.0)
                self._sock.settimeout(0.5)
                self._reader = threading.Thread(target=self._receive_loop, daemon=True)
                self._reader.start()
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.25)
        raise TimeoutError(f"Could not connect to streaming Whisper at {self.host}:{self.port}: {last_error}")

    def _receive_loop(self) -> None:
        assert self._sock is not None
        buffer = b""
        while not self._stop.is_set():
            try:
                data = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                decoded = _decode_transcript_line(line.decode("utf-8", errors="replace"))
                if decoded:
                    with self._lock:
                        self._transcripts.append(decoded)
        if buffer.strip():
            decoded = _decode_transcript_line(buffer.decode("utf-8", errors="replace"))
            if decoded:
                with self._lock:
                    self._transcripts.append(decoded)

    def all_transcripts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._transcripts)

    def send_audio(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        if self._sock is None:
            raise RuntimeError("StreamingWhisperClient is not connected.")
        pcm_bytes = mono_audio_to_pcm16(audio, sr)
        chunk_size = max(1, int(STREAM_BYTES_PER_SECOND * self.packet_ms / 1000.0))
        start = time.perf_counter()
        for offset in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[offset:offset + chunk_size]
            self._sock.sendall(chunk)
            if self.realtime:
                time.sleep(len(chunk) / float(STREAM_BYTES_PER_SECOND))
        elapsed = time.perf_counter() - start
        audio_sec = len(pcm_bytes) / float(STREAM_BYTES_PER_SECOND)
        return elapsed, audio_sec

    def close(self, final_wait_sec: float) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        time.sleep(max(0.0, final_wait_sec))
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)


def start_streaming_whisper_server(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        str(args.python_executable),
        str(SIMULSTREAMING_ROOT / "simulstreaming_whisper_server.py"),
        "--host",
        args.streaming_host,
        "--port",
        str(args.streaming_port),
        "--model_path",
        str(args.streaming_model_path),
        "--min-chunk-size",
        str(args.streaming_min_chunk_size),
        "--audio_max_len",
        str(args.streaming_audio_max_len),
        "--lan",
        args.language,
        "--task",
        "transcribe",
        "-l",
        args.streaming_log_level,
    ]
    if args.streaming_warmup_file is not None:
        command.extend(["--warmup-file", str(args.streaming_warmup_file)])
    return subprocess.Popen(
        command,
        cwd=str(SIMULSTREAMING_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


@dataclass(frozen=True)
class EvalChunk:
    fileid: int
    chunk_index: int
    mic_path: Path
    mic_file: str
    source_count: int
    gt_source_doas: str
    chunk_start_sample: int
    chunk_end_sample: int


@dataclass
class HarkChunkRecord:
    fileid: int
    chunk_index: int
    source_wav_file: str
    chunk_wav: str
    output_dir: str
    output_prefix: str
    command: str
    returncode: int
    elapsed_sec: float
    stdout_log: str
    stderr_log: str
    separated_wav_count: int


@dataclass
class SceneWer:
    fileid: int
    speaker_id: int
    speaker_role: str
    source_wav_file: str
    source_count: int
    gt_source_doa: Optional[int]
    gt_source_doas: str
    gt_text_file: str
    selected_candidate_index: int
    selected_enhanced_wav: str
    candidate_count: int
    reference_text: str
    hypothesis_text: str
    wer: Optional[float]
    no_insertion_wer: Optional[float]
    edit_distance: Optional[int]
    substitutions: Optional[int]
    deletions: Optional[int]
    insertions: Optional[int]
    ref_words: Optional[int]
    whisper_sec: float


def format_gt_doas(doas: Sequence[Optional[int]]) -> str:
    return ",".join("NA" if doa is None else str(doa) for doa in doas)


def unique_respeaker_recording_files(
    recordings_dir: Path,
    max_fileids: int,
    max_files: int,
    source_count_filter: int,
) -> List[Path]:
    all_files = sorted(
        recordings_dir.glob("fileid_*_sources_*.wav"),
        key=lambda p: (parse_respeaker_recording_name(p)[0], p.name),
    )
    if source_count_filter > 0:
        all_files = [
            path
            for path in all_files
            if parse_respeaker_recording_name(path)[1] == source_count_filter
        ]
    if max_fileids > 0:
        selected_fileids = sorted({parse_respeaker_recording_name(path)[0] for path in all_files})[:max_fileids]
        selected_fileid_set = set(selected_fileids)
        all_files = [path for path in all_files if parse_respeaker_recording_name(path)[0] in selected_fileid_set]
    if max_files > 0:
        return all_files[:max_files]
    return all_files


def build_respeaker_eval_chunks(recording_files: Sequence[Path]) -> List[EvalChunk]:
    chunks: List[EvalChunk] = []

    for wav_path in recording_files:
        fileid, source_count, doas = parse_respeaker_recording_name(wav_path)
        info = sf.info(str(wav_path))
        chunks.append(
            EvalChunk(
                fileid=fileid,
                chunk_index=0,
                mic_path=wav_path,
                mic_file=f"{wav_path.stem}_full.wav",
                source_count=source_count,
                gt_source_doas=format_gt_doas(doas),
                chunk_start_sample=0,
                chunk_end_sample=info.frames,
            )
        )
    return chunks


def load_eval_chunk_audio(
    chunk: EvalChunk,
    target_sr: int,
) -> Tuple[np.ndarray, int, int]:
    wav_tc, sr = sf.read(
        str(chunk.mic_path),
        start=chunk.chunk_start_sample,
        stop=chunk.chunk_end_sample,
        always_2d=True,
    )
    wav_tc, sr = resample_multichannel_audio(wav_tc.astype(np.float32), sr, target_sr)
    valid_samples = wav_tc.shape[0]
    return wav_tc, sr, valid_samples


def load_offset_respeaker_texts(text_dir: Path, fileid_offset: int) -> Dict[int, Path]:
    texts: Dict[int, Path] = {}
    for text_path in sorted(text_dir.glob("*.txt")):
        if text_path.stem.isdigit():
            fileid = int(text_path.stem) - fileid_offset
            if fileid >= 0:
                texts[fileid] = text_path
    return texts


def source_doa(gt_source_doas: str, speaker_id: int) -> Optional[int]:
    values: List[Optional[int]] = []
    for raw in gt_source_doas.split(","):
        raw = raw.strip()
        values.append(None if not raw or raw.upper() == "NA" else int(raw) % 360)
    if speaker_id < 1 or speaker_id > len(values):
        return None
    return values[speaker_id - 1]


def scene_output_dir(args: argparse.Namespace, chunk: EvalChunk) -> Path:
    return args.out_dir / "official_hark_outputs" / f"fileid_{chunk.fileid}" / "full_scene"


def scene_output_prefix(args: argparse.Namespace, chunk: EvalChunk) -> Path:
    return scene_output_dir(args, chunk) / f"hark_fileid_{chunk.fileid}_full"


def chunk_wav_path(args: argparse.Namespace, chunk: EvalChunk) -> Path:
    return args.out_dir / "hark_input_full_scenes" / f"fileid_{chunk.fileid}" / "full_scene.wav"


def find_hark_wavs(args: argparse.Namespace, chunk: EvalChunk) -> List[Path]:
    out_dir = scene_output_dir(args, chunk)
    if not out_dir.exists():
        return []
    return sort_hark_outputs([path for path in out_dir.glob(args.official_output_glob) if path.is_file()])


def sort_hark_outputs(wavs: Sequence[Path]) -> List[Path]:
    def key(path: Path) -> Tuple[int, str]:
        match = re.search(r"_(\d+)\.wav$", path.name)
        return (int(match.group(1)) if match else 10000, path.name)

    return sorted(wavs, key=key)


def write_chunk_wav(args: argparse.Namespace, chunk: EvalChunk) -> Tuple[Path, int, int]:
    wav_tc, sr, valid_samples = load_eval_chunk_audio(chunk, args.sample_rate)
    wav_path = chunk_wav_path(args, chunk)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav_path), wav_tc, sr)
    return wav_path, sr, valid_samples


def patch_hark_network_for_chunk(args: argparse.Namespace, chunk: EvalChunk, wav_path: Path) -> Path:
    if not args.hark_network.is_file():
        raise FileNotFoundError(f"HARK network file not found: {args.hark_network}")

    out_dir = scene_output_dir(args, chunk)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_network = out_dir / f"runtime_fileid_{chunk.fileid}_full.n"
    output_basename = str(scene_output_prefix(args, chunk)) + "_"

    text = args.hark_network.read_text(encoding="utf-8")
    shebang = ""
    if text.startswith("#!"):
        first_newline = text.find("\n")
        shebang = text[: first_newline + 1]
        text = text[first_newline + 1 :]

    root = ET.fromstring(text)
    if args.tf_zip is not None and not args.tf_zip.is_file():
        raise FileNotFoundError(f"Transfer-function zip not found: {args.tf_zip}")

    for node in root.iter("Node"):
        node_type = node.attrib.get("type")
        for parameter in node.findall("Parameter"):
            name = parameter.attrib.get("name")
            if node_type == "Constant" and name == "VALUE":
                parameter.set("value", str(wav_path))
            elif node_type == "SaveWavePCM" and name == "BASENAME":
                parameter.set("value", output_basename)
            elif name in {"A_MATRIX", "TF_CONJ_FILENAME"} and args.tf_zip is not None:
                parameter.set("value", str(args.tf_zip))

    xml_text = ET.tostring(root, encoding="unicode")
    runtime_network.write_text(shebang + '<?xml version="1.0"?>\n' + xml_text + "\n", encoding="utf-8")
    return runtime_network


def render_hark_command(args: argparse.Namespace, chunk: EvalChunk, wav_path: Path) -> List[str]:
    runtime_network = patch_hark_network_for_chunk(args, chunk, wav_path)
    out_dir = scene_output_dir(args, chunk)
    output_prefix = scene_output_prefix(args, chunk)
    values = {
        "runner": str(args.hark_runner),
        "network": str(runtime_network),
        "input_wav": str(wav_path),
        "output_prefix": str(output_prefix),
        "output_dir": str(out_dir),
        "fileid": str(chunk.fileid),
        "chunk_index": str(chunk.chunk_index),
        "mic_stem": chunk.mic_path.stem,
    }
    return shlex.split(args.hark_command_template.format(**values))


def run_hark_on_chunks(args: argparse.Namespace, chunks: Sequence[EvalChunk]) -> List[HarkChunkRecord]:
    runner_path = shutil.which(args.hark_runner)
    if runner_path is None and not Path(args.hark_runner).is_file():
        raise FileNotFoundError(
            f"HARK runner not found on PATH: {args.hark_runner}. "
            "Run this inside the Ubuntu/WSL environment where official HARK is installed."
        )

    logs_dir = args.out_dir / "official_hark_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    records: List[HarkChunkRecord] = []

    for chunk in tqdm(chunks, desc="Official-HARK", unit="scene"):
        out_dir = scene_output_dir(args, chunk)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = find_hark_wavs(args, chunk)
        chunk_wav, _, _ = write_chunk_wav(args, chunk)
        if args.skip_existing and existing:
            records.append(
                HarkChunkRecord(
                    fileid=chunk.fileid,
                    chunk_index=chunk.chunk_index,
                    source_wav_file=str(chunk.mic_path),
                    chunk_wav=str(chunk_wav),
                    output_dir=str(out_dir),
                    output_prefix=str(scene_output_prefix(args, chunk)),
                    command="SKIPPED_EXISTING",
                    returncode=0,
                    elapsed_sec=0.0,
                    stdout_log="",
                    stderr_log="",
                    separated_wav_count=len(existing),
                )
            )
            continue

        command = render_hark_command(args, chunk, chunk_wav)
        stdout_log = logs_dir / f"hark_fileid_{chunk.fileid}_full.stdout.log"
        stderr_log = logs_dir / f"hark_fileid_{chunk.fileid}_full.stderr.log"
        start = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - start
        stdout_log.write_text(proc.stdout or "", encoding="utf-8")
        stderr_log.write_text(proc.stderr or "", encoding="utf-8")
        wavs = find_hark_wavs(args, chunk)
        records.append(
            HarkChunkRecord(
                fileid=chunk.fileid,
                chunk_index=chunk.chunk_index,
                source_wav_file=str(chunk.mic_path),
                chunk_wav=str(chunk_wav),
                output_dir=str(out_dir),
                output_prefix=str(scene_output_prefix(args, chunk)),
                command=" ".join(shlex.quote(part) for part in command),
                returncode=int(proc.returncode),
                elapsed_sec=elapsed,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
                separated_wav_count=len(wavs),
            )
        )

    manifest_csv = args.out_dir / "official_hark_full_scene_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HarkChunkRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))
    return records


def group_chunks_by_fileid(chunks: Iterable[EvalChunk]) -> Dict[int, List[EvalChunk]]:
    grouped: Dict[int, List[EvalChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.fileid, []).append(chunk)
    for file_chunks in grouped.values():
        file_chunks.sort(key=lambda item: item.chunk_index)
    return grouped


def get_hark_candidate_wavs(
    args: argparse.Namespace,
    fileid: int,
    chunks: Sequence[EvalChunk],
) -> List[Path]:
    if len(chunks) != 1:
        raise ValueError(f"Full-scene mode expects exactly one HARK input for fileid {fileid}.")
    return find_hark_wavs(args, chunks[0])


def stream_transcribe_audio(wav_path: Path, args: argparse.Namespace) -> Tuple[str, float]:
    audio, _ = load_mono_audio(wav_path, target_sr=args.sample_rate)
    if args.streaming_trailing_silence_sec > 0:
        silence = np.zeros(int(round(args.streaming_trailing_silence_sec * args.sample_rate)), dtype=np.float32)
        audio = np.concatenate([audio, silence])

    client = StreamingWhisperClient(
        host=args.streaming_host,
        port=args.streaming_port,
        packet_ms=args.stream_packet_ms,
        realtime=args.stream_realtime,
        connect_timeout=args.stream_connect_timeout,
    )
    start = time.perf_counter()
    client.connect()
    try:
        client.send_audio(audio, args.sample_rate)
    finally:
        client.close(args.stream_final_wait)
    sec = time.perf_counter() - start
    return transcript_text(client.all_transcripts()), sec


def choose_best_candidate_for_text(
    reference_text: str,
    candidate_transcripts: Sequence[Tuple[int, Path, str, float]],
) -> Tuple[int, Path, str, float, Tuple[float, float, int, int, int, int, int]]:
    candidates = []
    for candidate_idx, wav_path, hyp_text, whisper_sec in candidate_transcripts:
        stats = word_error_stats(reference_text, hyp_text)
        wer_value, no_insert_wer, edits, substitutions, deletions, insertions, ref_words = stats
        candidates.append(
            (
                wer_value,
                no_insert_wer,
                edits,
                candidate_idx,
                wav_path,
                hyp_text,
                whisper_sec,
                stats,
            )
        )
    if not candidates:
        raise ValueError("No candidate transcripts available for oracle WER selection.")
    _, _, _, candidate_idx, wav_path, hyp_text, whisper_sec, stats = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    return candidate_idx, wav_path, hyp_text, whisper_sec, stats


def evaluate_hark_outputs(
    args: argparse.Namespace,
    recording_files: Sequence[Path],
    chunks: Sequence[EvalChunk],
) -> List[SceneWer]:
    server_proc: Optional[subprocess.Popen] = None
    if args.streaming_mode == "managed":
        if not args.streaming_model_path.is_file():
            raise FileNotFoundError(f"SimulStreaming Whisper model not found: {args.streaming_model_path}")
        print("Starting SimulStreaming Whisper server once...")
        server_proc = start_streaming_whisper_server(args)

    dominant_texts = load_offset_respeaker_texts(args.dominant_text_dir, args.dominant_text_fileid_offset)
    minor_texts = load_offset_respeaker_texts(args.minor_text_dir, args.minor_text_fileid_offset)
    chunks_by_fileid = group_chunks_by_fileid(chunks)
    recording_by_fileid = {parse_respeaker_recording_name(path)[0]: path for path in recording_files}

    records: List[SceneWer] = []
    skipped = {
        "no_candidates": 0,
        "missing_spk1_text": 0,
        "missing_spk2_text": 0,
    }

    try:
        for fileid, file_chunks in tqdm(chunks_by_fileid.items(), desc="Eval-HARK-SimulStreaming", unit="scene"):
            candidate_wavs = get_hark_candidate_wavs(args, fileid, file_chunks)
            if not candidate_wavs:
                skipped["no_candidates"] += 1
                continue

            candidate_transcripts: List[Tuple[int, Path, str, float]] = []
            for candidate_idx, wav_path in enumerate(candidate_wavs):
                hyp_text, whisper_sec = stream_transcribe_audio(wav_path, args)
                candidate_transcripts.append((candidate_idx, wav_path, hyp_text, whisper_sec))

            first_chunk = file_chunks[0]
            source_wav = recording_by_fileid.get(fileid, first_chunk.mic_path)
            for speaker_id, role, text_map in (
                (1, "dominant_spk1", dominant_texts),
                (2, "non_dominant_spk2", minor_texts),
            ):
                text_path = text_map.get(fileid)
                if text_path is None:
                    skipped[f"missing_spk{speaker_id}_text"] += 1
                    continue

                reference_text = text_path.read_text(encoding="utf-8").strip()
                candidate_idx, wav_path, hyp_text, whisper_sec, stats = choose_best_candidate_for_text(
                    reference_text,
                    candidate_transcripts,
                )
                (
                    wer_value,
                    no_insert_wer,
                    edits,
                    substitutions,
                    deletions,
                    insertions,
                    ref_words,
                ) = stats
                records.append(
                    SceneWer(
                        fileid=fileid,
                        speaker_id=speaker_id,
                        speaker_role=role,
                        source_wav_file=source_wav.name,
                        source_count=first_chunk.source_count,
                        gt_source_doa=source_doa(first_chunk.gt_source_doas, speaker_id),
                        gt_source_doas=first_chunk.gt_source_doas,
                        gt_text_file=str(text_path),
                        selected_candidate_index=candidate_idx,
                        selected_enhanced_wav=str(wav_path),
                        candidate_count=len(candidate_wavs),
                        reference_text=reference_text,
                        hypothesis_text=hyp_text,
                        wer=wer_value,
                        no_insertion_wer=no_insert_wer,
                        edit_distance=edits,
                        substitutions=substitutions,
                        deletions=deletions,
                        insertions=insertions,
                        ref_words=ref_words,
                        whisper_sec=whisper_sec,
                    )
                )
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    details_csv = args.out_dir / f"hark_locsep_whisper_{args.whisper_model}_scene_wer.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(SceneWer.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in records:
            writer.writerow(asdict(row))

    summary = build_summary(args, records, skipped, len(recording_files))
    summary_json = args.out_dir / f"hark_locsep_whisper_{args.whisper_model}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_csv = args.out_dir / f"hark_locsep_whisper_{args.whisper_model}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n===== REAL RESPEAKER HARK LOC+SEP WER SUMMARY =====")
    print(f"Evaluated scene-speaker WER items: {summary['evaluated_scene_speaker_items']}")
    print(
        "spk1 dominant: "
        f"corpus WER={summary['spk1_corpus_wer']:.4f}, "
        f"corpus no-insertion WER={summary['spk1_corpus_no_insertion_wer']:.4f}, "
        f"mean WER={summary['spk1_mean_scene_wer']:.4f}, "
        f"mean no-insertion WER={summary['spk1_mean_scene_no_insertion_wer']:.4f}, "
        f"items={summary['spk1_items']}"
    )
    print(
        "spk2 non-dominant: "
        f"corpus WER={summary['spk2_corpus_wer']:.4f}, "
        f"corpus no-insertion WER={summary['spk2_corpus_no_insertion_wer']:.4f}, "
        f"mean WER={summary['spk2_mean_scene_wer']:.4f}, "
        f"mean no-insertion WER={summary['spk2_mean_scene_no_insertion_wer']:.4f}, "
        f"items={summary['spk2_items']}"
    )
    print(f"Selection: {summary['selection']}")
    print(f"Saved scene WER: {details_csv}")
    print(f"Saved summary: {summary_json}")
    return records


def aggregate_speaker(rows: Sequence[SceneWer], speaker_id: int) -> Dict[str, float | int]:
    valid = [row for row in rows if row.speaker_id == speaker_id and row.wer is not None]
    ref_words = sum(int(row.ref_words or 0) for row in valid)
    edits = sum(int(row.edit_distance or 0) for row in valid)
    substitutions = sum(int(row.substitutions or 0) for row in valid)
    deletions = sum(int(row.deletions or 0) for row in valid)
    insertions = sum(int(row.insertions or 0) for row in valid)
    return {
        "items": len(valid),
        "corpus_wer": (edits / ref_words) if ref_words else 0.0,
        "corpus_no_insertion_wer": ((substitutions + deletions) / ref_words) if ref_words else 0.0,
        "mean_scene_wer": float(np.mean([row.wer for row in valid])) if valid else 0.0,
        "mean_scene_no_insertion_wer": (
            float(np.mean([row.no_insertion_wer for row in valid if row.no_insertion_wer is not None]))
            if valid
            else 0.0
        ),
        "edits": edits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ref_words": ref_words,
    }


def build_summary(
    args: argparse.Namespace,
    rows: Sequence[SceneWer],
    skipped: Dict[str, int],
    recording_count: int,
) -> Dict[str, object]:
    spk1 = aggregate_speaker(rows, 1)
    spk2 = aggregate_speaker(rows, 2)
    return {
        "respeaker_dir": str(args.respeaker_dir),
        "hark_network": str(args.hark_network),
        "tf_zip": str(args.tf_zip),
        "hark_runner": args.hark_runner,
        "mode": args.mode,
        "hark_input_mode": "full_recording",
        "hark_concat_trim": "none",
        "recording_count": recording_count,
        "official_output_glob": args.official_output_glob,
        "selection": "scene_level_oracle_wer_over_full_hark_candidate_streams",
        "asr_backend": "SimulStreaming Whisper server",
        "whisper_model_label": args.whisper_model,
        "streaming_model_path": str(args.streaming_model_path),
        "streaming_mode": args.streaming_mode,
        "streaming_host": args.streaming_host,
        "streaming_port": args.streaming_port,
        "streaming_min_chunk_size": args.streaming_min_chunk_size,
        "streaming_audio_max_len": args.streaming_audio_max_len,
        "stream_realtime": args.stream_realtime,
        "stream_packet_ms": args.stream_packet_ms,
        "stream_final_wait": args.stream_final_wait,
        "streaming_trailing_silence_sec": args.streaming_trailing_silence_sec,
        "dominant_text_dir": str(args.dominant_text_dir),
        "dominant_text_fileid_offset": args.dominant_text_fileid_offset,
        "minor_text_dir": str(args.minor_text_dir),
        "minor_text_fileid_offset": args.minor_text_fileid_offset,
        "evaluated_scene_speaker_items": len(rows),
        "skipped_no_candidates": skipped["no_candidates"],
        "missing_spk1_text": skipped["missing_spk1_text"],
        "missing_spk2_text": skipped["missing_spk2_text"],
        "spk1_items": spk1["items"],
        "spk1_corpus_wer": spk1["corpus_wer"],
        "spk1_corpus_no_insertion_wer": spk1["corpus_no_insertion_wer"],
        "spk1_mean_scene_wer": spk1["mean_scene_wer"],
        "spk1_mean_scene_no_insertion_wer": spk1["mean_scene_no_insertion_wer"],
        "spk1_substitutions": spk1["substitutions"],
        "spk1_deletions": spk1["deletions"],
        "spk1_insertions": spk1["insertions"],
        "spk1_ref_words": spk1["ref_words"],
        "spk2_items": spk2["items"],
        "spk2_corpus_wer": spk2["corpus_wer"],
        "spk2_corpus_no_insertion_wer": spk2["corpus_no_insertion_wer"],
        "spk2_mean_scene_wer": spk2["mean_scene_wer"],
        "spk2_mean_scene_no_insertion_wer": spk2["mean_scene_no_insertion_wer"],
        "spk2_substitutions": spk2["substitutions"],
        "spk2_deletions": spk2["deletions"],
        "spk2_insertions": spk2["insertions"],
        "spk2_ref_words": spk2["ref_words"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official HARK loc+sep on real ReSpeaker recordings for spk1 and spk2 WER."
    )
    parser.add_argument(
        "--mode",
        choices=("run_official", "eval_official", "both"),
        default="both",
        help="Run HARK, evaluate existing HARK full-scene outputs, or do both.",
    )
    parser.add_argument(
        "--respeaker_dir",
        type=Path,
        default=REAL_RESPEAKER_ROOT / "mic",
        help="Folder containing real ReSpeaker multichannel mixture wav files.",
    )
    parser.add_argument(
        "--respeaker_source_count",
        type=int,
        default=0,
        help="Filter ReSpeaker wavs by filename source count; 0 means use all.",
    )
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument(
        "--hark_network",
        type=Path,
        default=BASELINE_ROOT / "HARK" / "loc+sep.n",
        help="Official HARK loc+sep .n file.",
    )
    parser.add_argument(
        "--tf_zip",
        type=Path,
        default=BASELINE_ROOT / "HARK" / "respeaker4_tf_5deg.zip",
        help="Transfer-function zip patched into HARK nodes.",
    )
    parser.add_argument("--hark_runner", type=str, default="batchflow")
    parser.add_argument(
        "--hark_command_template",
        type=str,
        default="{runner} {network} {input_wav} {output_prefix}",
        help=(
            "HARK command template. Fields: {runner}, {network}, {input_wav}, "
            "{output_prefix}, {output_dir}, {fileid}, {chunk_index}, {mic_stem}."
        ),
    )
    parser.add_argument("--official_output_glob", type=str, default="*.wav")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--whisper_model", type=str, default="small", help="Label used in output filenames.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_items", type=int, default=0, help="Limit total scene fileids; 0 means all.")
    parser.add_argument("--max_files", type=int, default=0, help="Optional raw wav-file cap; 0 means all.")
    parser.add_argument(
        "--streaming_mode",
        choices=("managed", "external"),
        default="managed",
        help="managed starts SimulStreaming Whisper; external connects to an already running server.",
    )
    parser.add_argument("--python_executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--streaming_host", type=str, default="localhost")
    parser.add_argument("--streaming_port", type=int, default=43001)
    parser.add_argument("--streaming_model_path", type=Path, default=SIMULSTREAMING_ROOT / "small.pt")
    parser.add_argument("--streaming_min_chunk_size", type=float, default=1.0)
    parser.add_argument("--streaming_audio_max_len", type=float, default=30.0)
    parser.add_argument("--streaming_log_level", type=str, default="WARNING")
    parser.add_argument("--streaming_warmup_file", type=Path, default=None)
    parser.add_argument("--stream_connect_timeout", type=float, default=120.0)
    parser.add_argument("--stream_packet_ms", type=int, default=100)
    parser.add_argument("--stream_realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stream_final_wait", type=float, default=2.0)
    parser.add_argument(
        "--streaming_trailing_silence_sec",
        type=float,
        default=1.0,
        help="Append this much silence to each candidate before closing the streaming connection.",
    )
    parser.add_argument(
        "--dominant_text_dir",
        type=Path,
        default=REAL_RESPEAKER_ROOT / "text" / "src1",
        help="Reference transcript folder for source 1 / dominant spk1.",
    )
    parser.add_argument(
        "--dominant_text_fileid_offset",
        type=int,
        default=100,
        help="Text filename offset: 100.txt maps to fileid_0 when this is 100.",
    )
    parser.add_argument(
        "--minor_text_dir",
        type=Path,
        default=REAL_RESPEAKER_ROOT / "text" / "src2",
        help="Reference transcript folder for source 2 / non-dominant spk2.",
    )
    parser.add_argument(
        "--minor_text_fileid_offset",
        type=int,
        default=200,
        help="Text filename offset: 200.txt maps to fileid_0 when this is 200.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.respeaker_dir.is_dir():
        raise FileNotFoundError(f"ReSpeaker folder not found: {args.respeaker_dir}")
    if not args.dominant_text_dir.is_dir():
        raise FileNotFoundError(f"Dominant spk1 text folder not found: {args.dominant_text_dir}")
    if not args.minor_text_dir.is_dir():
        raise FileNotFoundError(f"Non-dominant spk2 text folder not found: {args.minor_text_dir}")
    if args.mode in {"run_official", "both"} and not args.hark_network.is_file():
        raise FileNotFoundError(f"HARK network file not found: {args.hark_network}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recording_files = unique_respeaker_recording_files(
        args.respeaker_dir,
        args.max_items,
        args.max_files,
        args.respeaker_source_count,
    )
    chunks = build_respeaker_eval_chunks(recording_files)

    print(f"ReSpeaker input folder: {args.respeaker_dir}")
    print(f"HARK network: {args.hark_network}")
    print(f"TF zip: {args.tf_zip}")
    print(f"Mode: {args.mode}")
    print("HARK input mode: full recording, no 4s chunking")
    print(f"Selected recordings: {len(recording_files)}")
    print(f"Selected full-scene HARK inputs: {len(chunks)}")
    print(f"spk1 text folder: {args.dominant_text_dir}, offset={args.dominant_text_fileid_offset}")
    print(f"spk2 text folder: {args.minor_text_dir}, offset={args.minor_text_fileid_offset}")

    if args.mode in {"run_official", "both"}:
        run_hark_on_chunks(args, chunks)
    if args.mode in {"eval_official", "both"}:
        evaluate_hark_outputs(args, recording_files, chunks)


if __name__ == "__main__":
    main()
