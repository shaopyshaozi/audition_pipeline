"""
Online-style SSL -> DSE -> streaming ASR realtime benchmark.

Loads IPDNet2 and DSENet once, then streams each saved multichannel 4 s wav
one scene at a time into a SimulStreaming Whisper server. For each scene:

1. Run SSL once on the representative multichannel mixture.
2. Convert SSL output to up to three DOAs.
3. Run DSENet once with a batch of three DOA inputs.
4. Select the loudest enhanced output and send it to streaming Whisper.

This script is timing-only. It does not require ground-truth text.
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
import torch
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL"
DSE_ROOT = MODELS_ROOT / "DSE"
SIMULSTREAMING_ROOT = MODELS_ROOT / "SimulStreaming"
DSENET_DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk_4s_full"

STREAM_SAMPLE_RATE = 16000
STREAM_BYTES_PER_SAMPLE = 2
STREAM_BYTES_PER_SECOND = STREAM_SAMPLE_RATE * STREAM_BYTES_PER_SAMPLE

sys.path.insert(0, str(SSL_ROOT))
from IPDnet2_3spk import OnlineSpatialNet  # noqa: E402
import Module_3spk as ssl_module  # noqa: E402
from utils_ import audiowu_high_array_geometry, forgetting_norm  # noqa: E402

sys.path.insert(0, str(DSE_ROOT))
from DOATrainer import TrainModule  # noqa: E402
from models.arch.DSENet import DSENet  # noqa: E402
from models.utils.metrics import recover_scale  # noqa: E402


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


def parse_chunk_index(path_or_name: Path | str) -> int:
    match = re.search(r"_(\d+)$", Path(path_or_name).stem)
    if not match:
        raise ValueError(f"Could not parse chunk index from: {path_or_name}")
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


def unique_mic_files(mic_dir: Path, max_fileids: int, max_files: int) -> List[Path]:
    all_files = sorted(
        mic_dir.glob("*.wav"),
        key=lambda p: (parse_fileid(p), parse_chunk_index(p), parse_doa(p), p.name),
    )
    if max_fileids > 0:
        selected_fileids = sorted({parse_fileid(path) for path in all_files})[:max_fileids]
        selected_fileid_set = set(selected_fileids)
        all_files = [path for path in all_files if parse_fileid(path) in selected_fileid_set]
    if max_files > 0:
        return all_files[:max_files]
    return all_files


def group_targets_by_scene_chunk(mic_files: Iterable[Path]) -> Dict[Tuple[int, int], List[Path]]:
    grouped: Dict[Tuple[int, int], List[Path]] = {}
    for path in mic_files:
        grouped.setdefault((parse_fileid(path), parse_chunk_index(path)), []).append(path)
    return grouped


def choose_representative_mic(target_paths: Sequence[Path]) -> Path:
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


def signal_rms(sig: np.ndarray) -> float:
    sig64 = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(sig64)) + 1e-12))


def select_loudest_enhanced(enhanced_batch: Sequence[np.ndarray]) -> Tuple[int, float]:
    if not enhanced_batch:
        raise ValueError("Cannot select loudest enhanced signal from an empty batch.")
    rms_values = [signal_rms(enhanced) for enhanced in enhanced_batch]
    selected_idx = int(np.argmax(rms_values))
    return selected_idx, rms_values[selected_idx]


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
    active = score < vad_th

    valid_angles = []
    valid_weights = []
    for t in range(azi.shape[0]):
        for k in range(azi.shape[1]):
            if active[t, k]:
                valid_angles.append(azi[t, k])
                valid_weights.append(1.0 / (score[t, k] + 1e-6))

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


class IPDNet2Inference(torch.nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device_name = device
        self.arch = OnlineSpatialNet(
            dim_input=8,
            dim_output=18,
            num_layers=8,
            dim_hidden=96,
            num_heads=4,
            kernel_size=(5, 3),
            conv_groups=(8, 8),
            norms=["LN", "LN", "GN", "LN", "LN", "LN"],
            dim_squeeze=8,
            num_freqs=256,
            attention="mamba(16,4)",
            rope=False,
            time_compression_layer=0,
            fre_compression_ratio=16,
            time_compression_ratio=5,
        )
        self.dostft = ssl_module.STFT(win_len=512, win_shift_ratio=0.625, nfft=512)
        self.fre_range_used = range(1, 257)

    def forward(self, mic_sig_batch: torch.Tensor) -> torch.Tensor:
        in_batch = self.data_preprocess_inference(mic_sig_batch)[0]
        return self.arch(in_batch)

    def data_preprocess_inference(self, mic_sig_batch: torch.Tensor, eps: float = 1e-6) -> List[torch.Tensor]:
        stft = self.dostft(signal=mic_sig_batch)
        stft_rebatch = stft.permute(0, 3, 1, 2).to(self.device_name)
        mag = torch.abs(stft_rebatch)
        mean_value = forgetting_norm(mag, sample_length=249)
        stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
        stft_rebatch_imag = torch.imag(stft_rebatch) / (mean_value + eps)
        real_imag_batch = torch.cat((stft_rebatch_real, stft_rebatch_imag), dim=1)
        return [real_imag_batch[:, :, self.fre_range_used, :]]


def load_ipdnet2(ckpt_path: Path, device: str) -> Tuple[IPDNet2Inference, torch.nn.Module]:
    model = IPDNet2Inference(device=device)
    ckpt = torch_load_checkpoint(ckpt_path, device)
    state_dict = ckpt.get("state_dict", ckpt)
    arch_state = {
        key.replace("arch.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("arch.")
    }
    missing, unexpected = model.arch.load_state_dict(arch_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected IPDNet2 checkpoint keys: {unexpected[:5]}")
    if missing:
        print(f"Warning: IPDNet2 missing {len(missing)} arch keys while loading checkpoint.")

    model.eval().to(device)

    use_mic_id = [2, 4, 6, 8]
    mic_location = audiowu_high_array_geometry()[use_mic_id]
    doa_decoder = ssl_module.PredDOA_Inference(
        mic_location=mic_location,
        max_track=3,
        max_num_sources=1,
        dev=device,
    )
    doa_decoder.eval().to(device)
    return model, doa_decoder


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
    model = TrainModule.load_from_checkpoint(str(ckpt_path), arch=arch, map_location=device)
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
    duration_sec: float
    audio_start_sec: float
    audio_end_sec: float
    predicted_doa_count: int
    predicted_doas: str
    selected_enhanced_index: int
    selected_doa: int
    selected_rms: float
    selected_enhanced_file: str
    gt_dominant_spk1_doa: Optional[int]
    selected_doa_error_deg: Optional[float]
    ipdnet2_sec: float
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
    ipdnet2_rtf: float
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
    audio_start_sec: float
    audio_end_sec: float
    audio_duration_sec: float
    gt_dominant_spk1_doa: Optional[int]
    gt_text_file: str
    selected_doas: str
    mean_selected_doa_error_deg: Optional[float]
    reference_text: str
    hypothesis_text: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IPDNet2 -> DSENet -> streaming Whisper realtime benchmark.")
    parser.add_argument("--mic_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "mic_4s")
    parser.add_argument("--clean_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "text")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "checkpoints" / "ipdnet2_23.ckpt")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_96.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results")
    parser.add_argument("--whisper_model", type=str, default="small", help="Label used in output filenames.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--vad_th", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--max_items", type=int, default=0, help="Limit total scene fileids for a quick test; 0 means all.")
    parser.add_argument("--max_files", type=int, default=0, help="Optional raw wav-file cap after fileid filtering; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save the selected loudest enhanced wav.")
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

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if args.streaming_mode == "managed" and not args.streaming_model_path.is_file():
        raise FileNotFoundError(f"Streaming Whisper model not found: {args.streaming_model_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_realtime_enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Streaming Whisper: {args.streaming_mode} {args.streaming_host}:{args.streaming_port}")
    print(f"Streaming Whisper model path: {args.streaming_model_path}")
    print(f"Streaming realtime sender: {args.stream_realtime}")
    print(f"DSENet batch size: {args.num_sources}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Dominant speaker clean folder: {args.clean_dir}")
    print(f"Dominant speaker text folder: {args.text_dir}")
    print("Loading IPDNet2 once...")
    ipd_model, doa_decoder = load_ipdnet2(args.ipd_ckpt, args.device)
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)

    server_proc: Optional[subprocess.Popen] = None
    if args.streaming_mode == "managed":
        print("Starting SimulStreaming Whisper server once...")
        server_proc = start_streaming_whisper_server(args)

    target_files = unique_mic_files(args.mic_dir, args.max_items, args.max_files)
    grouped = group_targets_by_scene_chunk(target_files)
    dominant_spk1_doas = load_dominant_spk1_doas(args.clean_dir)
    dominant_spk1_texts = load_dominant_spk1_texts(args.text_dir)
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique scene-chunk groups: {len(grouped)}")
    print(f"Dominant spk1 DOA references: {len(dominant_spk1_doas)}")
    print(f"Dominant spk1 text references: {len(dominant_spk1_texts)}")

    scene_results: List[SceneTiming] = []
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
        for (fileid, chunk_index), target_paths in tqdm(
            sorted(grouped.items()),
            desc="Realtime",
            unit="chunk",
        ):
            mic_path = choose_representative_mic(target_paths)
            wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
            duration_sec = wav_tc.shape[0] / float(sr)
            mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)
            noisy_ct = torch.from_numpy(wav_tc.T.copy())

            def run_ssl():
                with torch.inference_mode():
                    pred_ipd = ipd_model(mic_batch)
                    return doa_decoder(pred_ipd)

            ssl_out, ipd_sec = elapsed_seconds(args.device, run_ssl)
            pred_doas = postprocess_doa_from_tensors(
                ssl_out["doa_est"],
                ssl_out["vad_est"],
                num_sources=args.num_sources,
                vad_th=args.vad_th,
            )

            if len(pred_doas) != args.num_sources:
                skipped_no_doa += 1
                print(
                    f"fileid={fileid}: expected {args.num_sources} SSL DOAs, "
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
                print(f"fileid={fileid}: DSENet produced no enhanced output, skipped.")
                continue

            selected_idx, selected_rms = select_loudest_enhanced(enhanced_batch)
            selected_doa = pred_doas[selected_idx]
            selected_save_name = (
                f"enhanced_fileid_{fileid}_chunk{chunk_index}_"
                f"pred{selected_doa}_idx{selected_idx}_loudest.wav"
            )
            enhanced_for_asr = enhanced_batch[selected_idx]
            gt_dominant_doa = dominant_spk1_doas.get(fileid)
            selected_doa_error = (
                circular_angle_error_deg(selected_doa, gt_dominant_doa)
                if gt_dominant_doa is not None
                else None
            )
            if gt_dominant_doa is None:
                missing_dominant_gt += 1

            if args.save_enhanced:
                sf.write(str(enhanced_dir / selected_save_name), enhanced_for_asr, sr)

            audio_start_sec = stream_client.total_audio_sec
            transcript_start_idx = stream_client.transcript_count()
            stream_paced_send_sec, chunk_audio_sec = stream_client.send_audio(enhanced_for_asr, sr)
            transcript_delta = stream_client.transcripts_since(transcript_start_idx)
            received_during_send_text = transcript_text(transcript_delta)

            frontend_compute_sec = ipd_sec + dse_sec
            audio_end_sec = audio_start_sec + chunk_audio_sec
            pipeline_wall_sec = time.perf_counter() - stream_start
            stream_audio_end_sec = stream_client.total_audio_sec
            pipeline_lag_sec = pipeline_wall_sec - stream_audio_end_sec
            realtime_ok = int(pipeline_lag_sec <= args.realtime_tolerance_sec)
            scene_results.append(
                SceneTiming(
                    fileid=fileid,
                    chunk_index=chunk_index,
                    mic_file=mic_path.name,
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
                    selected_doa_error_deg=selected_doa_error,
                    ipdnet2_sec=ipd_sec,
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
                    ipdnet2_rtf=ipd_sec / duration_sec,
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
        text_path = dominant_spk1_texts.get(fileid)
        reference_text = text_path.read_text(encoding="utf-8").strip() if text_path is not None else ""
        sample_wer: Optional[float] = None
        edit_distance: Optional[int] = None
        ref_words: Optional[int] = None
        if text_path is None:
            missing_text += 1
        else:
            sample_wer, edit_distance, ref_words = wer(reference_text, hypothesis_text)
            total_edits += edit_distance
            total_ref_words += ref_words

        doa_errors = [row.selected_doa_error_deg for row in rows if row.selected_doa_error_deg is not None]
        scene_wer_results.append(
            SceneWer(
                fileid=fileid,
                chunk_count=len(rows),
                chunk_indices=",".join(str(row.chunk_index) for row in rows),
                chunk_mic_files="|".join(row.mic_file for row in rows),
                audio_start_sec=rows[0].audio_start_sec,
                audio_end_sec=rows[-1].audio_end_sec,
                audio_duration_sec=sum(row.duration_sec for row in rows),
                gt_dominant_spk1_doa=dominant_spk1_doas.get(fileid),
                gt_text_file=text_path.name if text_path is not None else "",
                selected_doas=",".join(str(row.selected_doa) for row in rows),
                mean_selected_doa_error_deg=float(np.mean(doa_errors)) if doa_errors else None,
                reference_text=reference_text,
                hypothesis_text=hypothesis_text,
                wer=sample_wer,
                edit_distance=edit_distance,
                ref_words=ref_words,
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

    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
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
        "dse_batch_size": args.num_sources,
        "selection": "loudest_enhanced_rms",
        "max_fileids": args.max_items,
        "max_files": args.max_files,
        "selected_mic_wav_entries": len(target_files),
        "unique_scene_chunk_groups": len(grouped),
        "unique_scene_fileids": len(rows_by_fileid),
        "dominant_spk1_doa_references": len(dominant_spk1_doas),
        "dominant_spk1_text_references": len(dominant_spk1_texts),
        "evaluated_chunks": len(scene_results),
        "evaluated_scene_wer_items": len(evaluated_wer_rows),
        "missing_scene_text": missing_text,
        "corpus_wer": corpus_wer,
        "mean_scene_wer": mean_scene_wer,
        "total_wer_edits": total_edits,
        "total_wer_ref_words": total_ref_words,
        "skipped_no_doa": skipped_no_doa,
        "missing_dominant_gt": missing_dominant_gt,
        "mean_selected_doa_error_deg": float(np.mean(selected_doa_errors)) if selected_doa_errors else 0.0,
        "median_selected_doa_error_deg": float(np.median(selected_doa_errors)) if selected_doa_errors else 0.0,
        "p95_selected_doa_error_deg": float(np.percentile(selected_doa_errors, 95)) if selected_doa_errors else 0.0,
        "mean_duration_sec": float(np.mean([r.duration_sec for r in scene_results])) if scene_results else 0.0,
        "mean_ipdnet2_sec": float(np.mean([r.ipdnet2_sec for r in scene_results])) if scene_results else 0.0,
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

    print("\n===== STREAMING REALTIME SUMMARY =====")
    print(f"Evaluated chunks: {summary['evaluated_chunks']}")
    print(f"Evaluated scene WER items: {summary['evaluated_scene_wer_items']}")
    print(f"Corpus WER: {summary['corpus_wer']:.4f}")
    print(f"Mean scene WER: {summary['mean_scene_wer']:.4f}")
    print(f"Final audio sent: {summary['final_audio_sec']:.3f}s")
    print(f"Final pipeline wall time: {summary['final_pipeline_wall_sec']:.3f}s")
    print(f"Final pipeline lag: {summary['final_pipeline_lag_sec']:.3f}s")
    print(f"Final cumulative RTF: {summary['final_cumulative_rtf']:.3f}")
    print(f"Realtime tolerance: {summary['realtime_tolerance_sec']:.3f}s")
    print(f"Final realtime ok: {summary['final_realtime_ok']}")
    print(f"Missing dominant spk1 DOA references: {summary['missing_dominant_gt']}")
    print(
        "Selected loudest DOA error vs spk1 GT: "
        f"mean={summary['mean_selected_doa_error_deg']:.2f} deg, "
        f"median={summary['median_selected_doa_error_deg']:.2f} deg, "
        f"p95={summary['p95_selected_doa_error_deg']:.2f} deg"
    )
    print(
        "Mean timing per scene: "
        f"IPDNet2={summary['mean_ipdnet2_sec']:.3f}s, "
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


if __name__ == "__main__":
    main()
