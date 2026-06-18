"""
Online-style streaming ASR baseline using clean spk1 audio directly.

This script streams saved 4-second clean speaker-1 chunks to a SimulStreaming
Whisper server. It is intended as a clean-speech upper-bound baseline for
comparing mixture audio and enhanced IPDNet2 -> DSENet streaming ASR results.

For each chunk:

1. Load a clean spk1 4-second chunk.
2. Convert it to mono if needed.
3. Send the clean audio to SimulStreaming Whisper.
4. Assign returned transcript segments to chunk intervals.

After all chunks are processed, chunk transcripts are concatenated by scene
fileid and scored against the ground-truth dominant speaker 1 text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import socket
import subprocess
import string
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


ONLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ONLINE_ROOT.parent.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
SIMULSTREAMING_ROOT = MODELS_ROOT / "SimulStreaming"
DSENET_DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk_4s_full"

STREAM_SAMPLE_RATE = 16000
STREAM_BYTES_PER_SAMPLE = 2
STREAM_BYTES_PER_SECOND = STREAM_SAMPLE_RATE * STREAM_BYTES_PER_SAMPLE


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def clean_asr_hypothesis_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"[<>]+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([(\"'])\s+", r"\1", text)
    text = re.sub(r"\s+([)\"'])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


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


def parse_chunk_index(path_or_name: Path | str) -> int:
    match = re.search(r"_(\d+)$", Path(path_or_name).stem)
    if not match:
        raise ValueError(f"Could not parse chunk index from: {path_or_name}")
    return int(match.group(1))


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=False)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        if wav.ndim == 1:
            wav = resample_poly(wav, up, down).astype(np.float32)
        else:
            wav = np.stack(
                [resample_poly(wav[:, ch], up, down).astype(np.float32) for ch in range(wav.shape[1])],
                axis=1,
            )
        sr = target_sr
    return wav, sr


def audio_to_mono(wav_tc: np.ndarray) -> np.ndarray:
    if wav_tc.ndim == 1:
        return wav_tc.astype(np.float32)
    return wav_tc.mean(axis=1).astype(np.float32)


def mono_audio_to_pcm16(audio: np.ndarray, sr: int, target_sr: int = STREAM_SAMPLE_RATE) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)
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


def transcript_segments_for_interval(
    segments: Sequence[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for item in segments:
        if "start" not in item:
            continue
        seg_start = float(item["start"])
        seg_end = float(item.get("end", seg_start))
        midpoint = (seg_start + seg_end) / 2.0
        if start_sec <= midpoint < end_sec:
            selected.append(dict(item))
    return selected


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
        start_time: float,
        connect_timeout: float,
    ):
        self.host = host
        self.port = port
        self.packet_ms = packet_ms
        self.realtime = realtime
        self.start_time = start_time
        self.connect_timeout = connect_timeout
        self.total_bytes_sent = 0
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._recv_buffer = b""
        self._transcripts: List[Dict[str, Any]] = []
        self._transcript_lock = threading.Lock()

    @property
    def total_audio_sec(self) -> float:
        return self.total_bytes_sent / float(STREAM_BYTES_PER_SECOND)

    def connect(self) -> None:
        deadline = time.perf_counter() + self.connect_timeout
        last_error: Optional[OSError] = None
        while time.perf_counter() < deadline:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=2.0)
                self._sock.settimeout(0.5)
                self._reader = threading.Thread(target=self._receive_loop, daemon=True)
                self._reader.start()
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        raise TimeoutError(f"Could not connect to streaming Whisper at {self.host}:{self.port}: {last_error}")

    def _receive_loop(self) -> None:
        assert self._sock is not None
        while not self._reader_stop.is_set():
            try:
                packet = self._sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not packet:
                break
            self._recv_buffer += packet
            while b"\n" in self._recv_buffer:
                raw_line, self._recv_buffer = self._recv_buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace")
                transcript = _decode_transcript_line(line)
                if transcript:
                    transcript["received_wall_sec"] = time.perf_counter() - self.start_time
                    with self._transcript_lock:
                        self._transcripts.append(transcript)

    def transcript_count(self) -> int:
        with self._transcript_lock:
            return len(self._transcripts)

    def transcripts_since(self, start_index: int) -> List[Dict[str, Any]]:
        with self._transcript_lock:
            return [dict(item) for item in self._transcripts[start_index:]]

    def all_transcripts(self) -> List[Dict[str, Any]]:
        with self._transcript_lock:
            return [dict(item) for item in self._transcripts]

    def send_audio(self, audio: np.ndarray, sr: int) -> Tuple[float, float]:
        if self._sock is None:
            raise RuntimeError("StreamingWhisperClient.connect() must be called before send_audio().")
        pcm_bytes = mono_audio_to_pcm16(audio, sr, target_sr=STREAM_SAMPLE_RATE)
        chunk_size = max(1, int(STREAM_BYTES_PER_SECOND * self.packet_ms / 1000.0))
        send_start = time.perf_counter()
        for offset in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[offset:offset + chunk_size]
            self._sock.sendall(chunk)
            self.total_bytes_sent += len(chunk)
            if self.realtime:
                expected_wall = self.total_audio_sec
                actual_wall = time.perf_counter() - self.start_time
                sleep_time = expected_wall - actual_wall
                if sleep_time > 0:
                    time.sleep(sleep_time)
        send_sec = time.perf_counter() - send_start
        audio_duration_sec = len(pcm_bytes) / float(STREAM_BYTES_PER_SECOND)
        return send_sec, audio_duration_sec

    def close(self, final_wait_sec: float) -> None:
        if self._sock is None:
            return
        if final_wait_sec > 0:
            time.sleep(final_wait_sec)
        try:
            self._sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        self._reader_stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None


def start_streaming_whisper_server(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        str(args.python_executable),
        str(SIMULSTREAMING_ROOT / "simulstreaming_whisper_server.py"),
        "--host",
        args.streaming_host,
        "--port",
        str(args.streaming_port),
        "--language",
        args.language,
        "--task",
        "transcribe",
        "--model_path",
        str(args.streaming_model_path),
        "--min-chunk-size",
        str(args.streaming_min_chunk_size),
        "--audio_max_len",
        str(args.streaming_audio_max_len),
        "--log-level",
        args.streaming_log_level,
    ]
    if args.streaming_warmup_file:
        command.extend(["--warmup-file", str(args.streaming_warmup_file)])
    return subprocess.Popen(
        command,
        cwd=str(SIMULSTREAMING_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def is_spk1(path: Path) -> bool:
    name = path.stem.lower()
    return "spk1" in name or "speaker1" in name


def unique_clean_files(clean_4s_dir: Path, max_fileids: int, max_files: int) -> List[Path]:
    all_files = sorted(
        (path for path in clean_4s_dir.glob("*.wav") if is_spk1(path)),
        key=lambda p: (parse_fileid(p), parse_chunk_index(p), parse_doa(p), p.name),
    )
    if max_fileids > 0:
        selected_fileids = sorted({parse_fileid(path) for path in all_files})[:max_fileids]
        selected_set = set(selected_fileids)
        all_files = [path for path in all_files if parse_fileid(path) in selected_set]
    if max_files > 0:
        return all_files[:max_files]
    return all_files


def group_targets_by_scene_chunk(clean_files: Iterable[Path]) -> Dict[Tuple[int, int], List[Path]]:
    grouped: Dict[Tuple[int, int], List[Path]] = {}
    for path in clean_files:
        grouped.setdefault((parse_fileid(path), parse_chunk_index(path)), []).append(path)
    return grouped


def choose_clean_chunk(target_paths: Sequence[Path]) -> Path:
    return sorted(target_paths, key=lambda p: (parse_doa(p), p.name))[0]


def load_dominant_spk1_doas(clean_dir: Path) -> Dict[int, int]:
    dominant_doas: Dict[int, int] = {}
    for clean_path in sorted(clean_dir.glob("clean_fileid_*_doa*_spk1.wav")):
        fileid = parse_fileid(clean_path)
        dominant_doas[fileid] = parse_doa(clean_path)
    return dominant_doas


def load_dominant_spk1_texts(text_dir: Path) -> Dict[int, Path]:
    dominant_texts: Dict[int, Path] = {}
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk1.txt")):
        fileid = parse_fileid(text_path)
        dominant_texts[fileid] = text_path
    return dominant_texts


@dataclass
class ChunkTiming:
    fileid: int
    chunk_index: int
    clean_file: str
    duration_sec: float
    input_channels: int
    input_audio_type: str
    audio_start_sec: float
    audio_end_sec: float
    gt_dominant_spk1_doa: Optional[int]
    stream_paced_send_sec: float
    chunk_audio_sec: float
    stream_audio_end_sec: float
    pipeline_wall_sec: float
    pipeline_lag_sec: float
    realtime_tolerance_sec: float
    realtime_ok: int
    cumulative_rtf: float
    stream_paced_send_rtf: float
    received_during_send_count: int
    received_during_send_text: str
    timestamp_assigned_count: int
    timestamp_assigned_text: str
    previous_chunk_transcript_text: str
    timestamp_assigned_start_sec: Optional[float]
    timestamp_assigned_end_sec: Optional[float]


@dataclass
class SceneWer:
    fileid: int
    chunk_count: int
    chunk_indices: str
    chunk_clean_files: str
    audio_start_sec: float
    audio_end_sec: float
    audio_duration_sec: float
    gt_dominant_spk1_doa: Optional[int]
    gt_text_file: str
    input_audio_type: str
    reference_text: str
    hypothesis_text: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    hypothesis_text_cleaned: str
    wer_cleaned: Optional[float]
    edit_distance_cleaned: Optional[int]
    ref_words_cleaned: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream clean spk1 audio directly to Whisper and score scene WER.")
    parser.add_argument("--clean_4s_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "clean_4s")
    parser.add_argument("--clean_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "text")
    parser.add_argument("--out_dir", type=Path, default=ONLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small", help="Label used in output filenames.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_items", type=int, default=0, help="Limit total scene fileids for a quick test; 0 means all.")
    parser.add_argument("--max_files", type=int, default=0, help="Optional raw wav-file cap after fileid filtering; 0 means all.")
    parser.add_argument(
        "--streaming_mode",
        choices=["managed", "external"],
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
        "--realtime_tolerance_sec",
        type=float,
        default=0.5,
        help="Allowed cumulative lag before marking realtime_ok=0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.clean_4s_dir.is_dir():
        raise FileNotFoundError(f"Clean 4s folder not found: {args.clean_4s_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if args.streaming_mode == "managed" and not args.streaming_model_path.is_file():
        raise FileNotFoundError(f"Streaming Whisper model not found: {args.streaming_model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_audio_type = "clean_spk1"

    print(f"Streaming Whisper: {args.streaming_mode} {args.streaming_host}:{args.streaming_port}")
    print(f"Streaming Whisper model path: {args.streaming_model_path}")
    print(f"Streaming realtime sender: {args.stream_realtime}")
    print(f"Input audio type: {input_audio_type}")
    print(f"Clean 4s input folder: {args.clean_4s_dir}")
    print(f"Dominant speaker clean folder: {args.clean_dir}")
    print(f"Dominant speaker text folder: {args.text_dir}")
    print(f"Output folder: {args.out_dir}")

    server_proc: Optional[subprocess.Popen] = None
    if args.streaming_mode == "managed":
        print("Starting SimulStreaming Whisper server once...")
        server_proc = start_streaming_whisper_server(args)

    target_files = unique_clean_files(args.clean_4s_dir, args.max_items, args.max_files)
    grouped = group_targets_by_scene_chunk(target_files)
    dominant_spk1_doas = load_dominant_spk1_doas(args.clean_dir)
    dominant_spk1_texts = load_dominant_spk1_texts(args.text_dir)
    print(f"Selected clean 4s wav entries: {len(target_files)}")
    print(f"Unique scene-chunk groups: {len(grouped)}")
    print(f"Dominant spk1 DOA references: {len(dominant_spk1_doas)}")
    print(f"Dominant spk1 text references: {len(dominant_spk1_texts)}")

    chunk_results: List[ChunkTiming] = []
    missing_dominant_gt = 0
    stream_start = time.perf_counter()
    stream_client = StreamingWhisperClient(
        host=args.streaming_host,
        port=args.streaming_port,
        packet_ms=args.stream_packet_ms,
        realtime=args.stream_realtime,
        start_time=stream_start,
        connect_timeout=args.stream_connect_timeout,
    )
    print("Connecting to streaming Whisper server...")
    try:
        stream_client.connect()
    except Exception:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        raise
    stream_start = time.perf_counter()
    stream_client.start_time = stream_start

    try:
        for (fileid, chunk_index), target_paths in tqdm(
            sorted(grouped.items()),
            desc="Clean streaming",
            unit="chunk",
        ):
            clean_path = choose_clean_chunk(target_paths)
            wav_tc, sr = load_audio(clean_path, target_sr=args.sample_rate)
            duration_sec = wav_tc.shape[0] / float(sr)
            clean_audio = audio_to_mono(wav_tc)

            gt_dominant_doa = dominant_spk1_doas.get(fileid)
            if gt_dominant_doa is None:
                missing_dominant_gt += 1

            audio_start_sec = stream_client.total_audio_sec
            transcript_start_idx = stream_client.transcript_count()
            stream_paced_send_sec, chunk_audio_sec = stream_client.send_audio(clean_audio, sr)
            transcript_delta = stream_client.transcripts_since(transcript_start_idx)
            received_during_send_text = transcript_text(transcript_delta)

            audio_end_sec = audio_start_sec + chunk_audio_sec
            pipeline_wall_sec = time.perf_counter() - stream_start
            stream_audio_end_sec = stream_client.total_audio_sec
            pipeline_lag_sec = pipeline_wall_sec - stream_audio_end_sec
            realtime_ok = int(pipeline_lag_sec <= args.realtime_tolerance_sec)
            chunk_results.append(
                ChunkTiming(
                    fileid=fileid,
                    chunk_index=chunk_index,
                    clean_file=clean_path.name,
                    duration_sec=duration_sec,
                    input_channels=wav_tc.shape[1] if wav_tc.ndim > 1 else 1,
                    input_audio_type=input_audio_type,
                    audio_start_sec=audio_start_sec,
                    audio_end_sec=audio_end_sec,
                    gt_dominant_spk1_doa=gt_dominant_doa,
                    stream_paced_send_sec=stream_paced_send_sec,
                    chunk_audio_sec=chunk_audio_sec,
                    stream_audio_end_sec=stream_audio_end_sec,
                    pipeline_wall_sec=pipeline_wall_sec,
                    pipeline_lag_sec=pipeline_lag_sec,
                    realtime_tolerance_sec=args.realtime_tolerance_sec,
                    realtime_ok=realtime_ok,
                    cumulative_rtf=pipeline_wall_sec / stream_audio_end_sec if stream_audio_end_sec > 0 else 0.0,
                    stream_paced_send_rtf=stream_paced_send_sec / duration_sec,
                    received_during_send_count=len(transcript_delta),
                    received_during_send_text=received_during_send_text,
                    timestamp_assigned_count=0,
                    timestamp_assigned_text="",
                    previous_chunk_transcript_text="",
                    timestamp_assigned_start_sec=None,
                    timestamp_assigned_end_sec=None,
                )
            )
    finally:
        stream_client.close(args.stream_final_wait)
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    all_transcripts = stream_client.all_transcripts()
    transcript_jsonl = args.out_dir / f"pipeline_streaming_{args.whisper_model}_clean_transcripts_4s_full.jsonl"
    with transcript_jsonl.open("w", encoding="utf-8") as f:
        for item in all_transcripts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for row in chunk_results:
        assigned_segments = transcript_segments_for_interval(
            all_transcripts,
            row.audio_start_sec,
            row.audio_end_sec,
        )
        row.timestamp_assigned_count = len(assigned_segments)
        row.timestamp_assigned_text = transcript_text(assigned_segments)
        row.timestamp_assigned_start_sec = (
            min(float(item["start"]) for item in assigned_segments)
            if assigned_segments
            else None
        )
        row.timestamp_assigned_end_sec = (
            max(float(item.get("end", item["start"])) for item in assigned_segments)
            if assigned_segments
            else None
        )

    rows_by_fileid: Dict[int, List[ChunkTiming]] = {}
    for row in chunk_results:
        rows_by_fileid.setdefault(row.fileid, []).append(row)

    scene_wer_results: List[SceneWer] = []
    missing_text = 0
    total_edits = 0
    total_ref_words = 0
    total_edits_cleaned = 0
    total_ref_words_cleaned = 0
    for fileid, rows in sorted(rows_by_fileid.items()):
        rows = sorted(rows, key=lambda r: r.chunk_index)
        previous_text = ""
        for row in rows:
            row.previous_chunk_transcript_text = previous_text
            previous_text = row.timestamp_assigned_text

        hypothesis_text = " ".join(
            row.timestamp_assigned_text.strip()
            for row in rows
            if row.timestamp_assigned_text.strip()
        )
        hypothesis_text_cleaned = clean_asr_hypothesis_text(hypothesis_text)
        text_path = dominant_spk1_texts.get(fileid)
        reference_text = text_path.read_text(encoding="utf-8").strip() if text_path is not None else ""
        sample_wer: Optional[float] = None
        edit_distance: Optional[int] = None
        ref_words: Optional[int] = None
        sample_wer_cleaned: Optional[float] = None
        edit_distance_cleaned: Optional[int] = None
        ref_words_cleaned: Optional[int] = None
        if text_path is None:
            missing_text += 1
        else:
            sample_wer, edit_distance, ref_words = wer(reference_text, hypothesis_text)
            sample_wer_cleaned, edit_distance_cleaned, ref_words_cleaned = wer(
                reference_text,
                hypothesis_text_cleaned,
            )
            total_edits += edit_distance
            total_ref_words += ref_words
            total_edits_cleaned += edit_distance_cleaned
            total_ref_words_cleaned += ref_words_cleaned

        scene_wer_results.append(
            SceneWer(
                fileid=fileid,
                chunk_count=len(rows),
                chunk_indices=",".join(str(row.chunk_index) for row in rows),
                chunk_clean_files="|".join(row.clean_file for row in rows),
                audio_start_sec=rows[0].audio_start_sec,
                audio_end_sec=rows[-1].audio_end_sec,
                audio_duration_sec=sum(row.duration_sec for row in rows),
                gt_dominant_spk1_doa=dominant_spk1_doas.get(fileid),
                gt_text_file=text_path.name if text_path is not None else "",
                input_audio_type=input_audio_type,
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                wer=sample_wer,
                edit_distance=edit_distance,
                ref_words=ref_words,
                hypothesis_text_cleaned=hypothesis_text_cleaned,
                wer_cleaned=sample_wer_cleaned,
                edit_distance_cleaned=edit_distance_cleaned,
                ref_words_cleaned=ref_words_cleaned,
            )
        )

    final_audio_sec = stream_client.total_audio_sec
    final_wall_sec = time.perf_counter() - stream_start
    final_lag_sec = final_wall_sec - final_audio_sec
    final_cumulative_rtf = final_wall_sec / final_audio_sec if final_audio_sec > 0 else 0.0

    details_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_clean_details_4s_full.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(ChunkTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in chunk_results:
            writer.writerow(asdict(row))

    scene_wer_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_clean_scene_wer_4s_full.csv"
    with scene_wer_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneWer.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_wer_results:
            writer.writerow(asdict(row))

    evaluated_wer_rows = [row for row in scene_wer_results if row.wer is not None]
    corpus_wer = (total_edits / total_ref_words) if total_ref_words > 0 else 0.0
    mean_scene_wer = float(np.mean([row.wer for row in evaluated_wer_rows])) if evaluated_wer_rows else 0.0
    evaluated_cleaned_wer_rows = [row for row in scene_wer_results if row.wer_cleaned is not None]
    corpus_wer_cleaned = (
        total_edits_cleaned / total_ref_words_cleaned
        if total_ref_words_cleaned > 0
        else 0.0
    )
    mean_scene_wer_cleaned = (
        float(np.mean([row.wer_cleaned for row in evaluated_cleaned_wer_rows]))
        if evaluated_cleaned_wer_rows
        else 0.0
    )

    summary = {
        "script": SCRIPT_STEM,
        "pipeline_variant": "clean_spk1_direct_streaming_asr",
        "clean_4s_dir": str(args.clean_4s_dir),
        "clean_dir": str(args.clean_dir),
        "text_dir": str(args.text_dir),
        "out_dir": str(args.out_dir),
        "whisper_model_label": args.whisper_model,
        "streaming_mode": args.streaming_mode,
        "streaming_host": args.streaming_host,
        "streaming_port": args.streaming_port,
        "streaming_model_path": str(args.streaming_model_path),
        "streaming_min_chunk_size": args.streaming_min_chunk_size,
        "streaming_audio_max_len": args.streaming_audio_max_len,
        "stream_realtime": args.stream_realtime,
        "stream_packet_ms": args.stream_packet_ms,
        "realtime_tolerance_sec": args.realtime_tolerance_sec,
        "input_audio_type": input_audio_type,
        "max_fileids": args.max_items,
        "max_files": args.max_files,
        "selected_clean_4s_wav_entries": len(target_files),
        "unique_scene_chunk_groups": len(grouped),
        "unique_scene_fileids": len(rows_by_fileid),
        "dominant_spk1_doa_references": len(dominant_spk1_doas),
        "dominant_spk1_text_references": len(dominant_spk1_texts),
        "evaluated_chunks": len(chunk_results),
        "evaluated_scene_wer_items": len(evaluated_wer_rows),
        "missing_scene_text": missing_text,
        "missing_dominant_gt": missing_dominant_gt,
        "corpus_wer": corpus_wer,
        "mean_scene_wer": mean_scene_wer,
        "total_wer_edits": total_edits,
        "total_wer_ref_words": total_ref_words,
        "corpus_wer_cleaned": corpus_wer_cleaned,
        "mean_scene_wer_cleaned": mean_scene_wer_cleaned,
        "total_wer_edits_cleaned": total_edits_cleaned,
        "total_wer_ref_words_cleaned": total_ref_words_cleaned,
        "mean_duration_sec": float(np.mean([r.duration_sec for r in chunk_results])) if chunk_results else 0.0,
        "mean_stream_paced_send_sec": float(np.mean([r.stream_paced_send_sec for r in chunk_results])) if chunk_results else 0.0,
        "mean_stream_paced_send_rtf": float(np.mean([r.stream_paced_send_rtf for r in chunk_results])) if chunk_results else 0.0,
        "median_pipeline_lag_sec": float(np.median([r.pipeline_lag_sec for r in chunk_results])) if chunk_results else 0.0,
        "p95_pipeline_lag_sec": float(np.percentile([r.pipeline_lag_sec for r in chunk_results], 95)) if chunk_results else 0.0,
        "final_audio_sec": final_audio_sec,
        "final_pipeline_wall_sec": final_wall_sec,
        "final_pipeline_lag_sec": final_lag_sec,
        "final_cumulative_rtf": final_cumulative_rtf,
        "realtime_ok_count": int(sum(r.realtime_ok for r in chunk_results)),
        "realtime_ok_rate": float(np.mean([r.realtime_ok for r in chunk_results])) if chunk_results else 0.0,
        "final_realtime_ok": int(final_lag_sec <= args.realtime_tolerance_sec) if final_audio_sec > 0 else 0,
        "stream_transcript_segments": len(all_transcripts),
        "timestamp_assigned_transcript_segments": int(sum(r.timestamp_assigned_count for r in chunk_results)),
    }

    summary_json = args.out_dir / f"pipeline_streaming_{args.whisper_model}_clean_summary_4s_full.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== CLEAN STREAMING ASR SUMMARY =====")
    print(f"Evaluated chunks: {summary['evaluated_chunks']}")
    print(f"Evaluated scene WER items: {summary['evaluated_scene_wer_items']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Mean scene WER: {summary['mean_scene_wer']:.4f}")
    print(f"Corpus WER cleaned: {summary['corpus_wer_cleaned']:.4f}")
    print(f"Mean scene WER cleaned: {summary['mean_scene_wer_cleaned']:.4f}")
    print(f"Final audio sent: {summary['final_audio_sec']:.3f}s")
    print(f"Final pipeline wall time: {summary['final_pipeline_wall_sec']:.3f}s")
    print(f"Final pipeline lag: {summary['final_pipeline_lag_sec']:.3f}s")
    print(f"Final cumulative RTF: {summary['final_cumulative_rtf']:.3f}")
    print(f"Realtime tolerance: {summary['realtime_tolerance_sec']:.3f}s")
    print(f"Final realtime ok: {summary['final_realtime_ok']}")
    print(f"Transcript segments received: {summary['stream_transcript_segments']}")
    print(f"Transcript segments timestamp-assigned: {summary['timestamp_assigned_transcript_segments']}")
    print(f"Saved streaming details: {details_csv}")
    print(f"Saved scene WER: {scene_wer_csv}")
    print(f"Saved transcripts: {transcript_jsonl}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
