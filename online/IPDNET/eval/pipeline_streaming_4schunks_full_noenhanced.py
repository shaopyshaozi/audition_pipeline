"""
Online-style raw ReSpeaker -> streaming ASR realtime baseline.

By default this script chops each long ReSpeaker mic-array recording into 4 s
chunks, applies ``--input_gain`` to the raw multichannel chunk, averages the mic
channels to mono, and streams that directly into a SimulStreaming Whisper server.
That gives a no-SSL/no-enhancement baseline for comparison against the full
IPDNET -> DSENet pipeline.

The old IPDNET -> DSENet -> loudest-enhanced path remains available with
``--frontend_mode full``.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = OFFLINE_ROOT.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL" / "IPDNET"
DSE_ROOT = MODELS_ROOT / "DSE"
SIMULSTREAMING_ROOT = MODELS_ROOT / "SimulStreaming"

STREAM_SAMPLE_RATE = 16000
STREAM_BYTES_PER_SAMPLE = 2
STREAM_BYTES_PER_SECOND = STREAM_SAMPLE_RATE * STREAM_BYTES_PER_SAMPLE
DEFAULT_CHUNK_SEC = 4.0

sys.path.insert(0, str(SSL_ROOT))
import FixedAarryIPDnet as ssl_model  # noqa: E402
import Module as ssl_module  # noqa: E402
from utils_ import forgetting_norm  # noqa: E402

sys.path.insert(0, str(DSE_ROOT))
from DOATrainer_3spk_myriad import TrainModule  # noqa: E402
from models.arch.DSENet import DSENet  # noqa: E402
from models.utils.metrics import recover_scale  # noqa: E402
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


def clean_asr_hypothesis_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b[A-Z][A-Z0-9_]*\b", " ", text)
    text = re.sub(r"\.{3,}|…+", " ", text)
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


def parse_gt_source_doas(gt_source_doas: str) -> List[Optional[int]]:
    doas: List[Optional[int]] = []
    for raw in gt_source_doas.split(","):
        raw = raw.strip()
        if not raw or raw.upper() == "NA":
            doas.append(None)
        else:
            doas.append(int(raw) % 360)
    return doas


def closest_gt_source(
    selected_doa: int,
    gt_source_doas: str,
) -> Tuple[Optional[int], Optional[float]]:
    valid_doas = [
        (idx, doa)
        for idx, doa in enumerate(parse_gt_source_doas(gt_source_doas), start=1)
        if doa is not None
    ]
    if not valid_doas:
        return None, None
    errors = [(idx, circular_angle_error_deg(selected_doa, doa)) for idx, doa in valid_doas]
    return min(errors, key=lambda item: item[1])


def parse_respeaker_recording_name(path_or_name: Path | str) -> Tuple[int, int, Tuple[Optional[int], ...]]:
    name = Path(path_or_name).name
    match = re.match(r"fileid_(\d+)_sources_(\d+)_([^_]+)_([^_]+)_([^_]+)\.wav$", name)
    if not match:
        raise ValueError(f"Could not parse Respeaker recording name: {path_or_name}")
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
    transcripts: Sequence[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
) -> List[Dict[str, Any]]:
    matched: List[Dict[str, Any]] = []
    for item in transcripts:
        if "start" not in item:
            continue
        seg_start = float(item.get("start", 0.0))
        seg_end = float(item.get("end", seg_start))
        midpoint = (seg_start + seg_end) / 2.0
        if start_sec <= midpoint < end_sec:
            matched.append(dict(item))
    return matched


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


@dataclass
class EvalChunk:
    fileid: int
    chunk_index: int
    mic_path: Path
    mic_file: str
    source_count: int
    gt_dominant_spk1_doa: Optional[int]
    gt_source_doas: str
    chunk_start_sample: int
    chunk_end_sample: int


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


def build_respeaker_eval_chunks(
    recording_files: Sequence[Path],
    chunk_sec: float,
    include_partial_chunk: bool,
) -> List[EvalChunk]:
    chunks: List[EvalChunk] = []
    if chunk_sec <= 0:
        raise ValueError(f"chunk_sec must be positive, got {chunk_sec}")

    for wav_path in recording_files:
        fileid, source_count, doas = parse_respeaker_recording_name(wav_path)
        info = sf.info(str(wav_path))
        chunk_samples = max(1, int(round(chunk_sec * info.samplerate)))
        chunk_index = 0
        for start_sample in range(0, info.frames, chunk_samples):
            end_sample = min(start_sample + chunk_samples, info.frames)
            if end_sample <= start_sample:
                continue
            if end_sample - start_sample < chunk_samples and not include_partial_chunk:
                continue
            chunks.append(
                EvalChunk(
                    fileid=fileid,
                    chunk_index=chunk_index,
                    mic_path=wav_path,
                    mic_file=f"{wav_path.stem}_chunk{chunk_index}.wav",
                    source_count=source_count,
                    gt_dominant_spk1_doa=doas[0] if doas else None,
                    gt_source_doas=format_gt_doas(doas),
                    chunk_start_sample=start_sample,
                    chunk_end_sample=end_sample,
                )
            )
            chunk_index += 1
    return chunks


def load_eval_chunk_audio(
    chunk: EvalChunk,
    target_sr: int,
    chunk_sec: float,
) -> Tuple[np.ndarray, int, int, float, float]:
    wav_tc, sr = sf.read(
        str(chunk.mic_path),
        start=chunk.chunk_start_sample,
        stop=chunk.chunk_end_sample,
        always_2d=True,
    )
    wav_tc, sr = resample_multichannel_audio(wav_tc.astype(np.float32), sr, target_sr)
    valid_samples = wav_tc.shape[0]
    target_samples = max(1, int(round(chunk_sec * sr)))
    if wav_tc.shape[0] < target_samples:
        pad = np.zeros((target_samples - wav_tc.shape[0], wav_tc.shape[1]), dtype=wav_tc.dtype)
        wav_tc = np.concatenate([wav_tc, pad], axis=0)

    info = sf.info(str(chunk.mic_path))
    chunk_start_sec = chunk.chunk_start_sample / float(info.samplerate)
    chunk_end_sec = chunk.chunk_end_sample / float(info.samplerate)
    return wav_tc, sr, valid_samples, chunk_start_sec, chunk_end_sec


def load_respeaker_texts(recordings_dir: Path) -> Dict[int, Path]:
    texts: Dict[int, Path] = {}
    for text_path in sorted(recordings_dir.glob("*.txt")):
        if text_path.stem.isdigit():
            texts[int(text_path.stem)] = text_path
    return texts


def signal_rms(sig: np.ndarray) -> float:
    sig64 = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(sig64)) + 1e-12))


def select_loudest_enhanced(enhanced_batch: Sequence[np.ndarray]) -> Tuple[int, float]:
    if not enhanced_batch:
        raise ValueError("Cannot select loudest enhanced signal from an empty batch.")
    rms_values = [signal_rms(enhanced) for enhanced in enhanced_batch]
    selected_idx = int(np.argmax(rms_values))
    return selected_idx, rms_values[selected_idx]


def select_raw_respeaker_audio(
    wav_tc: np.ndarray,
    valid_samples: int,
    raw_channel: int,
) -> Tuple[np.ndarray, str, float]:
    raw_tc = wav_tc[:valid_samples]
    if raw_channel >= 0:
        if raw_channel >= raw_tc.shape[1]:
            raise ValueError(
                f"Requested raw ReSpeaker channel {raw_channel}, "
                f"but chunk only has {raw_tc.shape[1]} channels."
            )
        raw_audio = raw_tc[:, raw_channel].astype(np.float32, copy=True)
        source_label = f"ch{raw_channel}"
    else:
        raw_audio = raw_tc.mean(axis=1).astype(np.float32)
        source_label = "mean_channels"
    return raw_audio, source_label, signal_rms(raw_audio)


def postprocess_doa_from_tensors(
    doa_est: torch.Tensor,
    vad_est: torch.Tensor,
    num_sources: int,
    vad_th: float,
) -> List[int]:
    """
    Cluster active SSL DOA points into exactly num_sources DOAs.

    This follows the original clustering approach, but does not remove duplicate
    rounded DOAs and does not discard weak/small clusters. That keeps a fixed
    three-DOA output for the downstream DSENet batch whenever enough active SSL
    points exist to form three clusters.
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

    x = noisy_ct.unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)
    doa = torch.tensor(doa_values, dtype=torch.long, device=device)
    width = torch.full((batch_size,), width_value, dtype=torch.long, device=device)

    with torch.inference_mode():
        yr_hat = dse_model.forward(x, doa, width)
        if dse_model.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(
                preds=yr_hat,
                mixture=x[:, dse_model.ref_channel, :],
                scale_src_together=True,
                norm_if_exceed_1=False,
            )

    return [yr_hat[idx, 0].detach().cpu().numpy().astype(np.float32) for idx in range(batch_size)]


@dataclass
class SceneTiming:
    fileid: int
    chunk_index: int
    mic_file: str
    source_count: int
    source_wav_file: str
    source_chunk_start_sec: float
    source_chunk_end_sec: float
    duration_sec: float
    audio_start_sec: float
    audio_end_sec: float
    predicted_doa_count: int
    predicted_doas: str
    selected_enhanced_index: int
    selected_doa: Optional[int]
    selected_rms: float
    selected_enhanced_file: str
    gt_dominant_spk1_doa: Optional[int]
    gt_source_doas: str
    selected_doa_error_deg: Optional[float]
    target_doa_correct: Optional[int]
    selected_closest_gt_source: Optional[int]
    selected_closest_gt_doa_error_deg: Optional[float]
    selected_closer_to_nontarget: int
    ipdnet_sec: float
    dsenet_sec: float
    frontend_compute_sec: float
    frontend_compute_rtf: float
    frontend_compute_margin_sec: float
    stream_paced_send_sec: float
    chunk_audio_sec: float
    stream_audio_end_sec: float
    pipeline_wall_sec: float
    pipeline_lag_sec: float
    realtime_tolerance_sec: float
    realtime_ok: int
    cumulative_rtf: float
    chunk_total_wall_sec: float
    ipdnet_rtf: float
    dsenet_rtf: float
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
    chunk_mic_files: str
    source_wav_files: str
    audio_start_sec: float
    audio_end_sec: float
    audio_duration_sec: float
    gt_dominant_spk1_doa: Optional[int]
    gt_source_doas: str
    gt_text_file: str
    concatenated_enhanced_file: str
    selected_doas: str
    mean_selected_doa_error_deg: Optional[float]
    target_doa_correct_chunk_count: int
    target_doa_correct_rate: Optional[float]
    scene_target_doa_correct: Optional[int]
    wrong_speaker_selection_chunk_count: int
    wrong_speaker_selection_rate: Optional[float]
    reference_text: str
    hypothesis_text: str
    wer: Optional[float]
    no_insertion_wer: Optional[float]
    edit_distance: Optional[int]
    substitutions: Optional[int]
    deletions: Optional[int]
    insertions: Optional[int]
    ref_words: Optional[int]
    hypothesis_text_cleaned: str
    wer_cleaned: Optional[float]
    no_insertion_wer_cleaned: Optional[float]
    edit_distance_cleaned: Optional[int]
    substitutions_cleaned: Optional[int]
    deletions_cleaned: Optional[int]
    insertions_cleaned: Optional[int]
    ref_words_cleaned: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw ReSpeaker input or IPDNET -> DSENet into streaming Whisper."
    )
    parser.add_argument("--respeaker_dir", type=Path, default=OFFLINE_ROOT / "Respeaker_recordings")
    parser.add_argument(
        "--respeaker_source_count",
        type=int,
        default=0,
        help="Filter Respeaker wavs by filename source count; 0 means use all source counts found.",
    )
    parser.add_argument("--chunk_sec", type=float, default=DEFAULT_CHUNK_SEC)
    parser.add_argument(
        "--include_partial_chunk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include and pad the final short Respeaker chunk.",
    )
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "last-v1.ckpt")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_v13_99.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results" / SCRIPT_STEM)
    parser.add_argument("--whisper_model", type=str, default="small", help="Label used in output filenames.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--frontend_mode",
        choices=["raw", "full"],
        default="raw",
        help=(
            "raw sends gain-adjusted ReSpeaker audio straight to SimulStreaming; "
            "full runs IPDNET -> DSENet -> loudest enhanced first."
        ),
    )
    parser.add_argument(
        "--raw_respeaker_channel",
        type=int,
        default=-1,
        help="Raw frontend channel to stream; -1 averages all ReSpeaker channels to mono.",
    )
    parser.add_argument("--vad_th", type=float, default=0.7)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--max_items", type=int, default=0, help="Limit total scene fileids for a quick test; 0 means all.")
    parser.add_argument("--max_files", type=int, default=0, help="Optional raw wav-file cap after fileid filtering; 0 means all.")
    parser.add_argument(
        "--save_enhanced",
        action="store_true",
        help="Save the audio sent to SimulStreaming: raw baseline audio in raw mode, enhanced audio in full mode.",
    )
    parser.add_argument(
        "--save_concatenated_enhanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one concatenated selected-enhanced wav per fileid.",
    )
    parser.add_argument("--concat_enhanced_subdir", type=str, default="concatenated_enhanced")
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
    parser.add_argument(
        "--target_doa_correct_threshold",
        type=float,
        default=30.0,
        help="Selected DoA is target-correct when it is within this many degrees of filename DoA1.",
    )
    parser.add_argument(
        "--input_gain",
        type=float,
        default=1.0,
        help="Linear gain applied to raw ReSpeaker audio before any frontend and before streaming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.respeaker_dir.is_dir():
        raise FileNotFoundError(f"Respeaker folder not found: {args.respeaker_dir}")
    if args.streaming_mode == "managed" and not args.streaming_model_path.is_file():
        raise FileNotFoundError(f"Streaming Whisper model not found: {args.streaming_model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    streamed_audio_dirname = (
        "pipeline_realtime_raw_respeaker"
        if args.frontend_mode == "raw"
        else "pipeline_realtime_enhanced"
    )
    enhanced_dir = args.out_dir / streamed_audio_dirname
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
    concatenated_enhanced_dir = args.out_dir / args.concat_enhanced_subdir
    if args.save_concatenated_enhanced:
        concatenated_enhanced_dir.mkdir(parents=True, exist_ok=True)

    target_files = unique_respeaker_recording_files(
        args.respeaker_dir,
        args.max_items,
        args.max_files,
        args.respeaker_source_count,
    )
    eval_chunks = build_respeaker_eval_chunks(
        target_files,
        chunk_sec=args.chunk_sec,
        include_partial_chunk=args.include_partial_chunk,
    )
    dominant_spk1_doas: Dict[int, int] = {}
    for path in target_files:
        fileid, _, doas = parse_respeaker_recording_name(path)
        if doas and doas[0] is not None:
            dominant_spk1_doas[fileid] = doas[0]
    dominant_spk1_texts = load_respeaker_texts(args.respeaker_dir)

    print(f"Device: {args.device}")
    print(f"Streaming Whisper: {args.streaming_mode} {args.streaming_host}:{args.streaming_port}")
    print(f"Streaming Whisper model path: {args.streaming_model_path}")
    print(f"Streaming realtime sender: {args.stream_realtime}")
    print(f"Frontend mode: {args.frontend_mode}")
    print(f"Input gain: {args.input_gain:g}")
    if args.frontend_mode == "raw":
        raw_source_label = (
            "mean of all channels"
            if args.raw_respeaker_channel < 0
            else f"channel {args.raw_respeaker_channel}"
        )
        print(f"Raw ReSpeaker source: {raw_source_label}")
        print("Skipping IPDNET and DSENet model loading.")
    else:
        source_count_label = (
            f"filename source count filtered to {args.respeaker_source_count}"
            if args.respeaker_source_count > 0
            else "filename source count"
        )
        print(f"DSENet batch size: {source_count_label}")
    print(f"Respeaker input folder: {args.respeaker_dir}")
    print(f"Respeaker chunk size: {args.chunk_sec:.3f}s")
    print(f"Include partial final chunk: {args.include_partial_chunk}")
    ipd_model: Optional[IPDNetInference] = None
    dse_model: Optional[TrainModule] = None
    if args.frontend_mode == "full":
        print("Loading IPDNET once...")
        ipd_model = load_ipdnet(args.ipd_ckpt, args.device)
        print("Loading DSENet once...")
        dse_model = load_dsenet(args.dse_ckpt, args.device)

    server_proc: Optional[subprocess.Popen] = None
    if args.streaming_mode == "managed":
        print("Starting SimulStreaming Whisper server once...")
        server_proc = start_streaming_whisper_server(args)

    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique scene-chunk groups: {len(eval_chunks)}")
    print(f"Dominant spk1 DOA references: {len(dominant_spk1_doas)}")
    print(f"Dominant spk1 text references: {len(dominant_spk1_texts)}")

    scene_results: List[SceneTiming] = []
    enhanced_chunks_by_fileid: Dict[int, List[Tuple[int, np.ndarray, int]]] = {}
    skipped_no_doa = 0
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
        for eval_chunk in tqdm(
            sorted(eval_chunks, key=lambda item: (item.fileid, item.chunk_index)),
            desc="Realtime",
            unit="chunk",
        ):
            fileid = eval_chunk.fileid
            chunk_index = eval_chunk.chunk_index
            wav_tc, sr, valid_samples, source_chunk_start_sec, source_chunk_end_sec = load_eval_chunk_audio(
                eval_chunk,
                target_sr=args.sample_rate,
                chunk_sec=args.chunk_sec,
            )
            wav_tc = np.clip(wav_tc * args.input_gain, -1.0, 1.0)
            duration_sec = valid_samples / float(sr)

            if args.frontend_mode == "raw":
                raw_prepare_start = time.perf_counter()
                enhanced_for_asr, raw_source_label, selected_rms = select_raw_respeaker_audio(
                    wav_tc,
                    valid_samples,
                    args.raw_respeaker_channel,
                )
                raw_prepare_sec = time.perf_counter() - raw_prepare_start
                pred_doas: List[int] = []
                selected_idx = -1
                selected_doa: Optional[int] = None
                selected_save_name = (
                    f"raw_respeaker_fileid_{fileid}_chunk{chunk_index}_"
                    f"{raw_source_label}_gain{args.input_gain:g}.wav"
                )
                ipd_sec = 0.0
                dse_sec = 0.0
                frontend_compute_sec = raw_prepare_sec
            else:
                assert ipd_model is not None
                assert dse_model is not None
                mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)
                noisy_ct = torch.from_numpy(wav_tc.T.copy())

                def run_ssl():
                    with torch.inference_mode():
                        return ipd_model(mic_batch)

                ssl_out, ipd_sec = elapsed_seconds(args.device, run_ssl)
                pred_doas = postprocess_doa_from_tensors(
                    ssl_out["doa_est"],
                    ssl_out["vad_est"],
                    num_sources=eval_chunk.source_count,
                    vad_th=args.vad_th,
                )

                if len(pred_doas) != eval_chunk.source_count:
                    skipped_no_doa += 1
                    print(
                        f"fileid={fileid} chunk={chunk_index}: expected {eval_chunk.source_count} SSL DOAs, "
                        f"got {len(pred_doas)}, skipped."
                    )
                    continue

                enhanced_batch, dse_sec = elapsed_seconds(
                    args.device,
                    lambda: enhance_doa_batch(
                        dse_model,
                        noisy_ct,
                        pred_doas,
                        args.width,
                        args.device,
                    ),
                )
                if args.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not enhanced_batch:
                    skipped_no_doa += 1
                    print(f"fileid={fileid} chunk={chunk_index}: DSENet produced no enhanced output, skipped.")
                    continue

                selected_idx, selected_rms = select_loudest_enhanced(enhanced_batch)
                selected_doa = pred_doas[selected_idx]
                selected_save_name = (
                    f"enhanced_fileid_{fileid}_chunk{chunk_index}_"
                    f"pred{selected_doa}_idx{selected_idx}_loudest.wav"
                )
                enhanced_for_asr = enhanced_batch[selected_idx][:valid_samples]
                frontend_compute_sec = ipd_sec + dse_sec

            gt_dominant_doa = eval_chunk.gt_dominant_spk1_doa
            selected_doa_error = (
                circular_angle_error_deg(selected_doa, gt_dominant_doa)
                if selected_doa is not None and gt_dominant_doa is not None
                else None
            )
            target_doa_correct = (
                int(selected_doa_error <= args.target_doa_correct_threshold)
                if selected_doa_error is not None
                else None
            )
            closest_source_index, closest_gt_error = (
                closest_gt_source(selected_doa, eval_chunk.gt_source_doas)
                if selected_doa is not None
                else (None, None)
            )
            selected_closer_to_nontarget = int(
                closest_source_index is not None
                and closest_source_index != 1
                and selected_doa_error is not None
                and closest_gt_error is not None
                and closest_gt_error < selected_doa_error
            )
            if gt_dominant_doa is None:
                missing_dominant_gt += 1

            if args.save_enhanced:
                sf.write(str(enhanced_dir / selected_save_name), enhanced_for_asr, sr)
            if args.save_concatenated_enhanced:
                enhanced_chunks_by_fileid.setdefault(fileid, []).append((chunk_index, enhanced_for_asr, sr))

            audio_start_sec = stream_client.total_audio_sec
            transcript_start_idx = stream_client.transcript_count()
            stream_paced_send_sec, chunk_audio_sec = stream_client.send_audio(enhanced_for_asr, sr)
            transcript_delta = stream_client.transcripts_since(transcript_start_idx)
            received_during_send_text = transcript_text(transcript_delta)

            audio_end_sec = audio_start_sec + chunk_audio_sec
            pipeline_wall_sec = time.perf_counter() - stream_start
            stream_audio_end_sec = stream_client.total_audio_sec
            pipeline_lag_sec = pipeline_wall_sec - stream_audio_end_sec
            realtime_ok = int(pipeline_lag_sec <= args.realtime_tolerance_sec)
            scene_results.append(
                SceneTiming(
                    fileid=fileid,
                    chunk_index=chunk_index,
                    mic_file=eval_chunk.mic_file,
                    source_count=eval_chunk.source_count,
                    source_wav_file=eval_chunk.mic_path.name,
                    source_chunk_start_sec=source_chunk_start_sec,
                    source_chunk_end_sec=source_chunk_end_sec,
                    duration_sec=duration_sec,
                    audio_start_sec=audio_start_sec,
                    audio_end_sec=audio_end_sec,
                    predicted_doa_count=len(pred_doas),
                    predicted_doas=",".join(str(doa) for doa in pred_doas),
                    selected_enhanced_index=selected_idx,
                    selected_doa=selected_doa,
                    selected_rms=selected_rms,
                    selected_enhanced_file=selected_save_name,
                    gt_dominant_spk1_doa=gt_dominant_doa,
                    gt_source_doas=eval_chunk.gt_source_doas,
                    selected_doa_error_deg=selected_doa_error,
                    target_doa_correct=target_doa_correct,
                    selected_closest_gt_source=closest_source_index,
                    selected_closest_gt_doa_error_deg=closest_gt_error,
                    selected_closer_to_nontarget=selected_closer_to_nontarget,
                    ipdnet_sec=ipd_sec,
                    dsenet_sec=dse_sec,
                    frontend_compute_sec=frontend_compute_sec,
                    frontend_compute_rtf=frontend_compute_sec / duration_sec,
                    frontend_compute_margin_sec=duration_sec - frontend_compute_sec,
                    stream_paced_send_sec=stream_paced_send_sec,
                    chunk_audio_sec=chunk_audio_sec,
                    stream_audio_end_sec=stream_audio_end_sec,
                    pipeline_wall_sec=pipeline_wall_sec,
                    pipeline_lag_sec=pipeline_lag_sec,
                    realtime_tolerance_sec=args.realtime_tolerance_sec,
                    realtime_ok=realtime_ok,
                    cumulative_rtf=pipeline_wall_sec / stream_audio_end_sec if stream_audio_end_sec > 0 else 0.0,
                    chunk_total_wall_sec=frontend_compute_sec + stream_paced_send_sec,
                    ipdnet_rtf=ipd_sec / duration_sec,
                    dsenet_rtf=dse_sec / duration_sec,
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
    transcript_jsonl = args.out_dir / f"pipeline_streaming_{args.whisper_model}_transcripts_4s_full.jsonl"
    with transcript_jsonl.open("w", encoding="utf-8") as f:
        for item in all_transcripts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    for row in scene_results:
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

    rows_by_fileid: Dict[int, List[SceneTiming]] = {}
    for row in scene_results:
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
        sample_no_insertion_wer: Optional[float] = None
        edit_distance: Optional[int] = None
        substitutions: Optional[int] = None
        deletions: Optional[int] = None
        insertions: Optional[int] = None
        ref_words: Optional[int] = None
        sample_wer_cleaned: Optional[float] = None
        sample_no_insertion_wer_cleaned: Optional[float] = None
        edit_distance_cleaned: Optional[int] = None
        substitutions_cleaned: Optional[int] = None
        deletions_cleaned: Optional[int] = None
        insertions_cleaned: Optional[int] = None
        ref_words_cleaned: Optional[int] = None
        if text_path is None:
            missing_text += 1
        else:
            (
                sample_wer,
                sample_no_insertion_wer,
                edit_distance,
                substitutions,
                deletions,
                insertions,
                ref_words,
            ) = word_error_stats(reference_text, hypothesis_text)
            (
                sample_wer_cleaned,
                sample_no_insertion_wer_cleaned,
                edit_distance_cleaned,
                substitutions_cleaned,
                deletions_cleaned,
                insertions_cleaned,
                ref_words_cleaned,
            ) = word_error_stats(
                reference_text,
                hypothesis_text_cleaned,
            )
            total_edits += edit_distance
            total_ref_words += ref_words
            total_edits_cleaned += edit_distance_cleaned
            total_ref_words_cleaned += ref_words_cleaned

        concatenated_enhanced_file = ""
        if args.save_concatenated_enhanced:
            enhanced_chunks = sorted(enhanced_chunks_by_fileid.get(fileid, []), key=lambda item: item[0])
            if enhanced_chunks:
                sample_rates = {sample_rate for _, _, sample_rate in enhanced_chunks}
                if len(sample_rates) != 1:
                    raise ValueError(f"Cannot concatenate fileid={fileid}: mixed sample rates {sample_rates}")
                concatenated_audio = np.concatenate([audio for _, audio, _ in enhanced_chunks])
                concatenated_prefix = "raw_respeaker" if args.frontend_mode == "raw" else "enhanced"
                concatenated_suffix = "direct" if args.frontend_mode == "raw" else "loudest"
                concatenated_enhanced_file = (
                    f"{concatenated_prefix}_fileid_{fileid}_concatenated_{concatenated_suffix}.wav"
                )
                sf.write(
                    str(concatenated_enhanced_dir / concatenated_enhanced_file),
                    concatenated_audio,
                    enhanced_chunks[0][2],
                )

        doa_errors = [row.selected_doa_error_deg for row in rows if row.selected_doa_error_deg is not None]
        target_correct_flags = [row.target_doa_correct for row in rows if row.target_doa_correct is not None]
        target_correct_count = int(sum(target_correct_flags))
        target_correct_rate = (
            target_correct_count / len(target_correct_flags)
            if target_correct_flags
            else None
        )
        scene_target_correct = (
            int(target_correct_rate >= 0.5)
            if target_correct_rate is not None
            else None
        )
        wrong_speaker_count = int(sum(row.selected_closer_to_nontarget for row in rows))
        wrong_speaker_rate = wrong_speaker_count / len(rows) if rows else None
        scene_wer_results.append(
            SceneWer(
                fileid=fileid,
                chunk_count=len(rows),
                chunk_indices=",".join(str(row.chunk_index) for row in rows),
                chunk_mic_files="|".join(row.mic_file for row in rows),
                source_wav_files="|".join(sorted({row.source_wav_file for row in rows})),
                audio_start_sec=rows[0].audio_start_sec,
                audio_end_sec=rows[-1].audio_end_sec,
                audio_duration_sec=sum(row.duration_sec for row in rows),
                gt_dominant_spk1_doa=dominant_spk1_doas.get(fileid),
                gt_source_doas=rows[0].gt_source_doas,
                gt_text_file=text_path.name if text_path is not None else "",
                concatenated_enhanced_file=concatenated_enhanced_file,
                selected_doas=",".join(
                    "" if row.selected_doa is None else str(row.selected_doa)
                    for row in rows
                ),
                mean_selected_doa_error_deg=float(np.mean(doa_errors)) if doa_errors else None,
                target_doa_correct_chunk_count=target_correct_count,
                target_doa_correct_rate=target_correct_rate,
                scene_target_doa_correct=scene_target_correct,
                wrong_speaker_selection_chunk_count=wrong_speaker_count,
                wrong_speaker_selection_rate=wrong_speaker_rate,
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                wer=sample_wer,
                no_insertion_wer=sample_no_insertion_wer,
                edit_distance=edit_distance,
                substitutions=substitutions,
                deletions=deletions,
                insertions=insertions,
                ref_words=ref_words,
                hypothesis_text_cleaned=hypothesis_text_cleaned,
                wer_cleaned=sample_wer_cleaned,
                no_insertion_wer_cleaned=sample_no_insertion_wer_cleaned,
                edit_distance_cleaned=edit_distance_cleaned,
                substitutions_cleaned=substitutions_cleaned,
                deletions_cleaned=deletions_cleaned,
                insertions_cleaned=insertions_cleaned,
                ref_words_cleaned=ref_words_cleaned,
            )
        )

    final_audio_sec = stream_client.total_audio_sec
    final_wall_sec = time.perf_counter() - stream_start
    final_lag_sec = final_wall_sec - final_audio_sec
    final_cumulative_rtf = final_wall_sec / final_audio_sec if final_audio_sec > 0 else 0.0

    details_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_details_1asr_4s_full.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    scene_wer_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_scene_wer_1asr_4s_full.csv"
    with scene_wer_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneWer.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_wer_results:
            writer.writerow(asdict(row))

    selected_doa_errors = [
        r.selected_doa_error_deg
        for r in scene_results
        if r.selected_doa_error_deg is not None
    ]
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

    def aggregate_wer_rows(rows: Sequence[SceneWer], cleaned: bool = False) -> Dict[str, float]:
        if cleaned:
            valid_rows = [row for row in rows if row.ref_words_cleaned is not None]
            substitutions_sum = sum(int(row.substitutions_cleaned or 0) for row in valid_rows)
            deletions_sum = sum(int(row.deletions_cleaned or 0) for row in valid_rows)
            insertions_sum = sum(int(row.insertions_cleaned or 0) for row in valid_rows)
            ref_words_sum = sum(int(row.ref_words_cleaned or 0) for row in valid_rows)
            wer_values = [row.wer_cleaned for row in valid_rows if row.wer_cleaned is not None]
            sd_wer_values = [
                row.no_insertion_wer_cleaned
                for row in valid_rows
                if row.no_insertion_wer_cleaned is not None
            ]
        else:
            valid_rows = [row for row in rows if row.ref_words is not None]
            substitutions_sum = sum(int(row.substitutions or 0) for row in valid_rows)
            deletions_sum = sum(int(row.deletions or 0) for row in valid_rows)
            insertions_sum = sum(int(row.insertions or 0) for row in valid_rows)
            ref_words_sum = sum(int(row.ref_words or 0) for row in valid_rows)
            wer_values = [row.wer for row in valid_rows if row.wer is not None]
            sd_wer_values = [
                row.no_insertion_wer
                for row in valid_rows
                if row.no_insertion_wer is not None
            ]

        return {
            "count": float(len(valid_rows)),
            "corpus_wer": (
                (substitutions_sum + deletions_sum + insertions_sum) / ref_words_sum
                if ref_words_sum > 0
                else 0.0
            ),
            "corpus_no_insertion_wer": (
                (substitutions_sum + deletions_sum) / ref_words_sum
                if ref_words_sum > 0
                else 0.0
            ),
            "mean_scene_wer": float(np.mean(wer_values)) if wer_values else 0.0,
            "mean_scene_no_insertion_wer": float(np.mean(sd_wer_values)) if sd_wer_values else 0.0,
            "substitutions": float(substitutions_sum),
            "deletions": float(deletions_sum),
            "insertions": float(insertions_sum),
            "ref_words": float(ref_words_sum),
        }

    wer_all = aggregate_wer_rows(scene_wer_results)
    wer_all_cleaned = aggregate_wer_rows(scene_wer_results, cleaned=True)
    doa_correct_rows = [row for row in scene_wer_results if row.scene_target_doa_correct == 1]
    doa_wrong_rows = [row for row in scene_wer_results if row.scene_target_doa_correct == 0]
    wer_doa_correct = aggregate_wer_rows(doa_correct_rows)
    wer_doa_wrong = aggregate_wer_rows(doa_wrong_rows)
    wer_doa_correct_cleaned = aggregate_wer_rows(doa_correct_rows, cleaned=True)
    wer_doa_wrong_cleaned = aggregate_wer_rows(doa_wrong_rows, cleaned=True)
    target_doa_evaluated_chunks = [r.target_doa_correct for r in scene_results if r.target_doa_correct is not None]
    target_doa_correct_chunks = int(sum(target_doa_evaluated_chunks))
    wrong_speaker_selection_chunks = int(sum(r.selected_closer_to_nontarget for r in scene_results))
    selection_label = (
        "raw_respeaker_mean_channels"
        if args.frontend_mode == "raw" and args.raw_respeaker_channel < 0
        else (
            f"raw_respeaker_channel_{args.raw_respeaker_channel}"
            if args.frontend_mode == "raw"
            else "loudest_enhanced_rms"
        )
    )

    summary = {
        "respeaker_dir": str(args.respeaker_dir),
        "respeaker_source_count_filter": args.respeaker_source_count,
        "chunk_sec": args.chunk_sec,
        "include_partial_chunk": args.include_partial_chunk,
        "frontend_mode": args.frontend_mode,
        "input_gain": args.input_gain,
        "raw_respeaker_channel": args.raw_respeaker_channel,
        "ipd_ckpt": str(args.ipd_ckpt),
        "dse_ckpt": str(args.dse_ckpt),
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
        "device": args.device,
        "dse_batch_size": "none_raw_frontend" if args.frontend_mode == "raw" else "source_count_from_filename",
        "selection": selection_label,
        "save_chunk_enhanced": args.save_enhanced,
        "streamed_audio_dir": str(enhanced_dir) if args.save_enhanced else "",
        "save_concatenated_enhanced": args.save_concatenated_enhanced,
        "concatenated_enhanced_dir": str(concatenated_enhanced_dir) if args.save_concatenated_enhanced else "",
        "max_fileids": args.max_items,
        "max_files": args.max_files,
        "selected_mic_wav_entries": len(target_files),
        "unique_scene_chunk_groups": len(eval_chunks),
        "unique_scene_fileids": len(rows_by_fileid),
        "dominant_spk1_doa_references": len(dominant_spk1_doas),
        "dominant_spk1_text_references": len(dominant_spk1_texts),
        "evaluated_chunks": len(scene_results),
        "evaluated_scene_wer_items": len(evaluated_wer_rows),
        "missing_scene_text": missing_text,
        "target_doa_correct_threshold": args.target_doa_correct_threshold,
        "target_doa_evaluated_chunks": len(target_doa_evaluated_chunks),
        "target_doa_correct_chunks": target_doa_correct_chunks,
        "target_doa_accuracy": (
            target_doa_correct_chunks / len(target_doa_evaluated_chunks)
            if target_doa_evaluated_chunks
            else 0.0
        ),
        "wrong_speaker_selection_chunks": wrong_speaker_selection_chunks,
        "wrong_speaker_selection_rate": (
            wrong_speaker_selection_chunks / len(scene_results)
            if scene_results
            else 0.0
        ),
        "scene_target_doa_correct_items": len(doa_correct_rows),
        "scene_target_doa_wrong_items": len(doa_wrong_rows),
        "corpus_wer": corpus_wer,
        "corpus_no_insertion_wer": wer_all["corpus_no_insertion_wer"],
        "mean_scene_wer": mean_scene_wer,
        "mean_scene_no_insertion_wer": wer_all["mean_scene_no_insertion_wer"],
        "total_wer_edits": total_edits,
        "total_wer_substitutions": wer_all["substitutions"],
        "total_wer_deletions": wer_all["deletions"],
        "total_wer_insertions": wer_all["insertions"],
        "total_wer_ref_words": total_ref_words,
        "corpus_wer_cleaned": corpus_wer_cleaned,
        "corpus_no_insertion_wer_cleaned": wer_all_cleaned["corpus_no_insertion_wer"],
        "mean_scene_wer_cleaned": mean_scene_wer_cleaned,
        "mean_scene_no_insertion_wer_cleaned": wer_all_cleaned["mean_scene_no_insertion_wer"],
        "total_wer_edits_cleaned": total_edits_cleaned,
        "total_wer_substitutions_cleaned": wer_all_cleaned["substitutions"],
        "total_wer_deletions_cleaned": wer_all_cleaned["deletions"],
        "total_wer_insertions_cleaned": wer_all_cleaned["insertions"],
        "total_wer_ref_words_cleaned": total_ref_words_cleaned,
        "corpus_wer_when_doa_correct": wer_doa_correct["corpus_wer"],
        "corpus_no_insertion_wer_when_doa_correct": wer_doa_correct["corpus_no_insertion_wer"],
        "mean_scene_wer_when_doa_correct": wer_doa_correct["mean_scene_wer"],
        "mean_scene_no_insertion_wer_when_doa_correct": wer_doa_correct["mean_scene_no_insertion_wer"],
        "corpus_wer_cleaned_when_doa_correct": wer_doa_correct_cleaned["corpus_wer"],
        "corpus_no_insertion_wer_cleaned_when_doa_correct": wer_doa_correct_cleaned["corpus_no_insertion_wer"],
        "mean_scene_wer_cleaned_when_doa_correct": wer_doa_correct_cleaned["mean_scene_wer"],
        "mean_scene_no_insertion_wer_cleaned_when_doa_correct": wer_doa_correct_cleaned["mean_scene_no_insertion_wer"],
        "corpus_wer_when_doa_wrong": wer_doa_wrong["corpus_wer"],
        "corpus_no_insertion_wer_when_doa_wrong": wer_doa_wrong["corpus_no_insertion_wer"],
        "mean_scene_wer_when_doa_wrong": wer_doa_wrong["mean_scene_wer"],
        "mean_scene_no_insertion_wer_when_doa_wrong": wer_doa_wrong["mean_scene_no_insertion_wer"],
        "corpus_wer_cleaned_when_doa_wrong": wer_doa_wrong_cleaned["corpus_wer"],
        "corpus_no_insertion_wer_cleaned_when_doa_wrong": wer_doa_wrong_cleaned["corpus_no_insertion_wer"],
        "mean_scene_wer_cleaned_when_doa_wrong": wer_doa_wrong_cleaned["mean_scene_wer"],
        "mean_scene_no_insertion_wer_cleaned_when_doa_wrong": wer_doa_wrong_cleaned["mean_scene_no_insertion_wer"],
        "skipped_no_doa": skipped_no_doa,
        "missing_dominant_gt": missing_dominant_gt,
        "mean_selected_doa_error_deg": float(np.mean(selected_doa_errors)) if selected_doa_errors else 0.0,
        "median_selected_doa_error_deg": float(np.median(selected_doa_errors)) if selected_doa_errors else 0.0,
        "p95_selected_doa_error_deg": float(np.percentile(selected_doa_errors, 95)) if selected_doa_errors else 0.0,
        "mean_duration_sec": float(np.mean([r.duration_sec for r in scene_results])) if scene_results else 0.0,
        "mean_ipdnet_sec": float(np.mean([r.ipdnet_sec for r in scene_results])) if scene_results else 0.0,
        "mean_dsenet_sec": float(np.mean([r.dsenet_sec for r in scene_results])) if scene_results else 0.0,
        "mean_frontend_compute_sec": float(np.mean([r.frontend_compute_sec for r in scene_results])) if scene_results else 0.0,
        "mean_frontend_compute_rtf": float(np.mean([r.frontend_compute_rtf for r in scene_results])) if scene_results else 0.0,
        "mean_frontend_compute_margin_sec": float(np.mean([r.frontend_compute_margin_sec for r in scene_results])) if scene_results else 0.0,
        "mean_stream_paced_send_sec": float(np.mean([r.stream_paced_send_sec for r in scene_results])) if scene_results else 0.0,
        "mean_chunk_total_wall_sec": float(np.mean([r.chunk_total_wall_sec for r in scene_results])) if scene_results else 0.0,
        "median_pipeline_lag_sec": float(np.median([r.pipeline_lag_sec for r in scene_results])) if scene_results else 0.0,
        "p95_pipeline_lag_sec": float(np.percentile([r.pipeline_lag_sec for r in scene_results], 95)) if scene_results else 0.0,
        "final_audio_sec": final_audio_sec,
        "final_pipeline_wall_sec": final_wall_sec,
        "final_pipeline_lag_sec": final_lag_sec,
        "final_cumulative_rtf": final_cumulative_rtf,
        "realtime_ok_count": int(sum(r.realtime_ok for r in scene_results)),
        "realtime_ok_rate": float(np.mean([r.realtime_ok for r in scene_results])) if scene_results else 0.0,
        "final_realtime_ok": int(final_lag_sec <= args.realtime_tolerance_sec) if final_audio_sec > 0 else 0,
        "stream_transcript_segments": len(all_transcripts),
        "timestamp_assigned_transcript_segments": int(sum(r.timestamp_assigned_count for r in scene_results)),
    }

    summary_json = args.out_dir / f"pipeline_streaming_{args.whisper_model}_summary_1asr_4s_full.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_summary_1asr_4s_full.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print("\n===== STREAMING REALTIME SUMMARY =====")
    print(f"Evaluated chunks: {summary['evaluated_chunks']}")
    print(f"Evaluated scene WER items: {summary['evaluated_scene_wer_items']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Corpus no-insertion WER: {summary['corpus_no_insertion_wer']:.4f}")
    print(f"Mean scene WER: {summary['mean_scene_wer']:.4f}")
    print(f"Corpus WER cleaned: {summary['corpus_wer_cleaned']:.4f}")
    print(f"Corpus no-insertion WER cleaned: {summary['corpus_no_insertion_wer_cleaned']:.4f}")
    print(f"Mean scene WER cleaned: {summary['mean_scene_wer_cleaned']:.4f}")
    if args.frontend_mode == "full":
        print(
            "Target DoA accuracy: "
            f"{summary['target_doa_accuracy']:.4f} "
            f"({summary['target_doa_correct_chunks']}/{summary['target_doa_evaluated_chunks']} chunks, "
            f"threshold={summary['target_doa_correct_threshold']:.1f} deg)"
        )
        print(f"Wrong-speaker selection rate: {summary['wrong_speaker_selection_rate']:.4f}")
        print(
            "WER when DoA correct: "
            f"corpus={summary['corpus_wer_when_doa_correct']:.4f}, "
            f"no_insert={summary['corpus_no_insertion_wer_when_doa_correct']:.4f}, "
            f"items={summary['scene_target_doa_correct_items']}"
        )
        print(
            "WER when DoA wrong: "
            f"corpus={summary['corpus_wer_when_doa_wrong']:.4f}, "
            f"no_insert={summary['corpus_no_insertion_wer_when_doa_wrong']:.4f}, "
            f"items={summary['scene_target_doa_wrong_items']}"
        )
    else:
        print("Target DoA accuracy: N/A for raw frontend baseline")
        print("Wrong-speaker selection rate: N/A for raw frontend baseline")
    print(f"Final audio sent: {summary['final_audio_sec']:.3f}s")
    print(f"Final pipeline wall time: {summary['final_pipeline_wall_sec']:.3f}s")
    print(f"Final pipeline lag: {summary['final_pipeline_lag_sec']:.3f}s")
    print(f"Final cumulative RTF: {summary['final_cumulative_rtf']:.3f}")
    print(f"Realtime tolerance: {summary['realtime_tolerance_sec']:.3f}s")
    print(f"Final realtime ok: {summary['final_realtime_ok']}")
    print(f"Missing dominant spk1 DOA references: {summary['missing_dominant_gt']}")
    if args.frontend_mode == "full":
        print(
            "Selected loudest DOA error vs spk1 GT: "
            f"mean={summary['mean_selected_doa_error_deg']:.2f} deg, "
            f"median={summary['median_selected_doa_error_deg']:.2f} deg, "
            f"p95={summary['p95_selected_doa_error_deg']:.2f} deg"
        )
    print(
        "Mean timing per scene: "
        f"IPDNET={summary['mean_ipdnet_sec']:.3f}s, "
        f"DSENet={summary['mean_dsenet_sec']:.3f}s, "
        f"frontend_compute={summary['mean_frontend_compute_sec']:.3f}s, "
        f"paced_send={summary['mean_stream_paced_send_sec']:.3f}s"
    )
    print(f"Transcript segments received: {summary['stream_transcript_segments']}")
    print(f"Transcript segments timestamp-assigned: {summary['timestamp_assigned_transcript_segments']}")
    print(f"Saved streaming details: {details_csv}")
    print(f"Saved scene WER: {scene_wer_csv}")
    print(f"Saved transcripts: {transcript_jsonl}")
    print(f"Saved summary: {summary_json}")
    print(f"Saved summary CSV: {summary_csv}")
    if args.save_enhanced:
        print(f"Saved streamed audio folder: {enhanced_dir}")
    if args.save_concatenated_enhanced:
        print(f"Saved concatenated streamed audio folder: {concatenated_enhanced_dir}")


if __name__ == "__main__":
    main()
