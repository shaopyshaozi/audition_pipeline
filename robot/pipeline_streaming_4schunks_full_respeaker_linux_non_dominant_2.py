"""
Fixed-DoA non-dominant speaker test pipeline.

This script records ReSpeaker audio in realtime, bypasses SSL completely, runs
DSENet for one constant target DoA, and sends that enhanced stream to
SimulStreaming Whisper. It is meant for the test case where a non-dominant
speaker gives the robot command from around the fixed DoA, while other speakers
remain active at other directions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent
SCRIPT_STEM = Path(__file__).stem
MODELS_ROOT = PROJECT_ROOT / "Models"
DSE_ROOT = MODELS_ROOT / "DSE"
SIMULSTREAMING_ROOT = MODELS_ROOT / "SimulStreaming"
DSENET_DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk_4s_full"
NON_DOMINANT_RESULTS_ROOT = OFFLINE_ROOT / "results" / "non_dominant"

STREAM_SAMPLE_RATE = 16000
STREAM_BYTES_PER_SAMPLE = 2
STREAM_BYTES_PER_SECOND = STREAM_SAMPLE_RATE * STREAM_BYTES_PER_SAMPLE

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


def clean_asr_hypothesis_text(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\.{3,}|…+", " ", text)
    text = re.sub(r"[<>]+", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([(\"'])\s+", r"\1", text)
    text = re.sub(r"\s+([)\"'])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def next_experiment_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"experiment_(\d+)", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return root / f"experiment_{max_index + 1}"


class StreamingWavWriter:
    def __init__(self, path: Path, sr: int):
        self.path = path
        self.sr = sr
        self._file: Optional[sf.SoundFile] = None
        self._channels: Optional[int] = None

    def write(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        channels = 1 if audio.ndim == 1 else int(audio.shape[1])
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._channels = channels
            self._file = sf.SoundFile(
                str(self.path),
                mode="w",
                samplerate=self.sr,
                channels=channels,
                subtype="PCM_16",
            )
        elif channels != self._channels:
            raise ValueError(f"Cannot append {channels}ch audio to {self._channels}ch wav: {self.path}")
        self._file.write(audio)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class LiveEventWriter:
    def __init__(
        self,
        paths: Sequence[Path],
        pretty_print: bool,
    ):
        seen: set[Path] = set()
        self.paths = []
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            self.paths.append(path)
        self.pretty_print = pretty_print
        self._files = []
        self._event_index = 0
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files.append(path.open("w", encoding="utf-8"))

    def emit(self, event: Dict[str, Any]) -> None:
        self._event_index += 1
        event = dict(event)
        event["event_index"] = self._event_index
        line = json.dumps(event, ensure_ascii=False)
        for file_obj in self._files:
            file_obj.write(line + "\n")
            file_obj.flush()
        if self.pretty_print:
            print(format_live_event(event), flush=True)

    def close(self) -> None:
        for file_obj in self._files:
            file_obj.close()
        self._files = []


def format_live_event(event: Dict[str, Any]) -> str:
    start = event.get("transcript_start_sec")
    end = event.get("transcript_end_sec")
    if start is None:
        start = event.get("audio_start_sec", 0.0)
    if end is None:
        end = event.get("audio_end_sec", start)
    doa = event.get("selected_doa")
    doa_label = "n/a" if doa is None or int(doa) < 0 else f"{int(doa):03d} deg"
    text = str(event.get("text") or "").strip()
    chunk_index = event.get("chunk_index", "?")
    return f"[{float(start):.2f}-{float(end):.2f}s | DoA {doa_label} | chunk {chunk_index}] {text}"


def parse_channel_indices(value: str) -> List[int]:
    channels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not channels:
        raise argparse.ArgumentTypeError("At least one ReSpeaker mic channel must be selected.")
    if len(set(channels)) != len(channels):
        raise argparse.ArgumentTypeError(f"Duplicate ReSpeaker mic channel in: {value}")
    return channels


def list_audio_devices() -> None:
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError("PyAudio is required to list ReSpeaker devices. Install pyaudio first.") from exc

    pa = pyaudio.PyAudio()
    try:
        print("Available PyAudio input devices:")
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            name = str(info.get("name", ""))
            channels = int(info.get("maxInputChannels", 0))
            rate = float(info.get("defaultSampleRate", 0.0))
            print(f"  [{idx}] {name} | input_channels={channels} | default_rate={rate:g}")
    finally:
        pa.terminate()


class ReSpeakerChunkSource:
    def __init__(
        self,
        sample_rate: int,
        input_channels: int,
        sample_width: int,
        input_device_index: Optional[int],
        frames_per_buffer: int,
        mic_channels: Sequence[int],
        chunk_seconds: float,
        queue_max_chunks: int,
    ):
        self.sample_rate = sample_rate
        self.input_channels = input_channels
        self.sample_width = sample_width
        self.input_device_index = input_device_index
        self.frames_per_buffer = frames_per_buffer
        self.mic_channels = list(mic_channels)
        self.chunk_samples = int(round(sample_rate * chunk_seconds))
        self.queue_max_chunks = queue_max_chunks
        self._pa = None
        self._stream = None
        self._pending = np.empty((0, len(self.mic_channels)), dtype=np.float32)
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=max(0, queue_max_chunks))
        self._stop_event = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._reader_error: Optional[BaseException] = None

        if self.sample_width != 2:
            raise ValueError("Only 16-bit ReSpeaker input is currently supported.")
        if self.chunk_samples <= 0:
            raise ValueError("--chunk_seconds must be positive.")
        if self.queue_max_chunks < 0:
            raise ValueError("--record_queue_max_chunks must be >= 0.")
        invalid = [ch for ch in self.mic_channels if ch < 0 or ch >= self.input_channels]
        if invalid:
            raise ValueError(
                f"Mic channel indices {invalid} are outside the input channel range 0..{self.input_channels - 1}."
            )

    def __enter__(self) -> "ReSpeakerChunkSource":
        try:
            import pyaudio
        except ImportError as exc:
            raise RuntimeError("PyAudio is required for live ReSpeaker capture. Install pyaudio first.") from exc

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=self.sample_rate,
            format=self._pa.get_format_from_width(self.sample_width),
            channels=self.input_channels,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.frames_per_buffer,
        )
        self._reader = threading.Thread(target=self._read_loop, name="respeaker-recorder", daemon=True)
        self._reader.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _put_chunk(self, chunk: np.ndarray) -> None:
        while not self._stop_event.is_set():
            try:
                self._queue.put(chunk, timeout=0.5)
                return
            except queue.Full:
                print("Warning: recording queue is full; processing is falling behind capture.")

    def _read_loop(self) -> None:
        assert self._stream is not None
        try:
            while not self._stop_event.is_set():
                data = self._stream.read(self.frames_per_buffer, exception_on_overflow=False)
                interleaved = np.frombuffer(data, dtype=np.int16)
                frames = interleaved.reshape(-1, self.input_channels)
                raw_mics = frames[:, self.mic_channels].astype(np.float32) / 32768.0
                self._pending = np.concatenate((self._pending, raw_mics), axis=0)

                while self._pending.shape[0] >= self.chunk_samples:
                    chunk = self._pending[: self.chunk_samples].copy()
                    self._pending = self._pending[self.chunk_samples :]
                    self._put_chunk(chunk)
        except BaseException as exc:
            self._reader_error = exc
        finally:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass

    def iter_chunks(self, max_chunks: int = 0) -> Iterator[np.ndarray]:
        if self._stream is None:
            raise RuntimeError("ReSpeakerChunkSource must be opened before reading chunks.")

        produced = 0
        while max_chunks <= 0 or produced < max_chunks:
            try:
                chunk = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._reader_error is not None:
                    raise RuntimeError("ReSpeaker recorder thread failed.") from self._reader_error
                continue
            if chunk is None:
                if self._reader_error is not None:
                    raise RuntimeError("ReSpeaker recorder thread failed.") from self._reader_error
                return
            produced += 1
            yield chunk


class TcpInt16ChunkSource:
    def __init__(
        self,
        host: str,
        port: int,
        sample_rate: int,
        channels: int,
        chunk_seconds: float,
    ):
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = int(round(sample_rate * chunk_seconds))
        self.chunk_bytes = self.chunk_samples * channels * STREAM_BYTES_PER_SAMPLE
        self._sock: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._peer = None

        if self.channels <= 0:
            raise ValueError("--tcp_channels must be positive.")
        if self.chunk_samples <= 0:
            raise ValueError("--chunk_seconds must be positive.")

    def __enter__(self) -> "TcpInt16ChunkSource":
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(1)
        print(f"Waiting for Windows ReSpeaker sender on {self.host}:{self.port}...")
        self._conn, self._peer = self._sock.accept()
        self._conn.settimeout(2.0)
        print(f"Connected audio sender: {self._peer}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _recv_exact(self, nbytes: int) -> bytes:
        if self._conn is None:
            raise RuntimeError("TCP audio source must be connected before reading chunks.")
        data = bytearray()
        while len(data) < nbytes:
            try:
                packet = self._conn.recv(nbytes - len(data))
            except socket.timeout:
                continue
            if not packet:
                break
            data.extend(packet)
        return bytes(data)

    def iter_chunks(self, max_chunks: int = 0) -> Iterator[np.ndarray]:
        produced = 0
        while max_chunks <= 0 or produced < max_chunks:
            data = self._recv_exact(self.chunk_bytes)
            if len(data) == 0:
                print("TCP audio sender closed the connection.")
                return
            if len(data) < self.chunk_bytes:
                print(f"Dropping incomplete TCP audio chunk: {len(data)} of {self.chunk_bytes} bytes.")
                return
            audio = np.frombuffer(data, dtype="<i2").reshape(-1, self.channels)
            produced += 1
            yield (audio.astype(np.float32) / 32768.0).copy()


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


def transcript_midpoint_sec(transcript: Dict[str, Any]) -> Optional[float]:
    if "start" not in transcript:
        return None
    start = float(transcript.get("start", 0.0))
    end = float(transcript.get("end", start))
    return (start + end) / 2.0


def find_scene_row_for_transcript(
    rows: Sequence["SceneTiming"],
    transcript: Dict[str, Any],
) -> Optional["SceneTiming"]:
    midpoint = transcript_midpoint_sec(transcript)
    if midpoint is None:
        return rows[-1] if rows else None
    for row in rows:
        if row.audio_start_sec <= midpoint < row.audio_end_sec:
            return row
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: min(
            abs(midpoint - row.audio_start_sec),
            abs(midpoint - row.audio_end_sec),
        ),
    )


def live_event_from_transcript(
    transcript: Dict[str, Any],
    row: "SceneTiming",
) -> Dict[str, Any]:
    text = str(transcript.get("text") or transcript.get("raw") or "").strip()
    transcript_start = float(transcript["start"]) if "start" in transcript else None
    transcript_end = (
        float(transcript.get("end", transcript["start"]))
        if "start" in transcript
        else None
    )
    predicted_doas = [
        int(item)
        for item in row.predicted_doas.split(",")
        if item.strip() and item.strip().lstrip("-").isdigit()
    ]
    selection_scores = [
        float(item)
        for item in row.selection_scores.split(",")
        if item.strip()
    ]
    return {
        "type": "asr_doa",
        "chunk_index": row.chunk_index,
        "audio_start_sec": row.audio_start_sec,
        "audio_end_sec": row.audio_end_sec,
        "transcript_start_sec": transcript_start,
        "transcript_end_sec": transcript_end,
        "text": text,
        "is_final": bool(transcript.get("is_final", False)),
        "selected_doa": row.selected_doa,
        "predicted_doas": predicted_doas,
        "selection_policy": row.selection_policy,
        "selection_scores": selection_scores,
        "selection_active_ratio": row.selection_active_ratio,
        "selected_enhanced_index": row.selected_enhanced_index,
        "asr_audio_kind": row.asr_audio_kind,
        "selected_enhanced_file": row.selected_enhanced_file,
        "received_wall_sec": transcript.get("received_wall_sec"),
        "whisper": transcript,
    }


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


def signal_rms(sig: np.ndarray) -> float:
    sig64 = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(sig64)) + 1e-12))


def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


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


def fallback_audio_for_skipped_chunk(wav_tc: np.ndarray, policy: str) -> np.ndarray:
    if policy == "silence":
        return np.zeros(wav_tc.shape[0], dtype=np.float32)
    if policy == "mixture":
        return wav_tc.mean(axis=1).astype(np.float32)
    raise ValueError(f"Unsupported skip ASR policy: {policy}")


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
    asr_audio_kind: str
    selected_enhanced_index: int
    selected_doa: int
    selected_rms: float
    selection_policy: str
    selection_scores: str
    selection_active_ratio: float
    selected_enhanced_file: str
    fixed_doa_sec: float
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
    fixed_doa_rtf: float
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
class ChunkTranscript:
    chunk_index: int
    mic_file: str
    audio_start_sec: float
    audio_end_sec: float
    duration_sec: float
    asr_audio_kind: str
    predicted_doa_count: int
    predicted_doas: str
    selected_doa: int
    pipeline_lag_sec: float
    timestamp_assigned_text: str
    hypothesis_text_cleaned: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record ReSpeaker audio and run fixed-DoA DSENet -> streaming Whisper."
    )
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_v13_99.ckpt")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output experiment directory. Defaults to the next experiment_i under --non_dominant_results_root.",
    )
    parser.add_argument(
        "--non_dominant_results_root",
        type=Path,
        default=NON_DOMINANT_RESULTS_ROOT,
        help="Root folder for fixed-DoA non-dominant speaker live experiments.",
    )
    parser.add_argument("--whisper_model", type=str, default="small", help="Label used in output filenames.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument(
        "--audio_source",
        choices=["pyaudio", "tcp"],
        default="pyaudio",
        help="pyaudio records directly in this process; tcp receives raw 4ch int16 audio from Windows.",
    )
    parser.add_argument("--tcp_host", type=str, default="0.0.0.0")
    parser.add_argument("--tcp_port", type=int, default=50007)
    parser.add_argument("--tcp_channels", type=int, default=4)
    parser.add_argument("--chunk_seconds", type=float, default=4.0)
    parser.add_argument(
        "--target_doa",
        type=int,
        default=0,
        help="Constant azimuth given to DSENet. Use 0 for the planned non-dominant command speaker.",
    )
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument(
        "--skip_asr_policy",
        choices=["silence", "mixture", "drop"],
        default="silence",
        help="What to send to ASR when DSENet cannot produce an enhanced chunk.",
    )
    parser.add_argument("--max_chunks", type=int, default=0, help="Limit live 4 s chunks for a quick test; 0 means run until Ctrl+C.")
    parser.add_argument("--max_items", type=int, default=0, help="Deprecated alias for --max_chunks when --max_chunks is 0.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save fixed-DoA enhanced wavs for each chunk.")
    parser.add_argument("--save_raw_chunks", action="store_true", help="Save captured 4-channel ReSpeaker chunks for debugging.")
    parser.add_argument(
        "--raw_input_gain",
        type=float,
        default=1.0,
        help="Linear gain applied to captured 4-channel chunks before frontend processing and raw chunk saving.",
    )
    parser.add_argument("--list_audio_devices", action="store_true", help="Print PyAudio input device indices and exit.")
    parser.add_argument("--respeaker_rate", type=int, default=16000)
    parser.add_argument("--respeaker_channels", type=int, default=6)
    parser.add_argument("--respeaker_width", type=int, default=2)
    parser.add_argument(
        "--respeaker_index",
        type=int,
        default=1,
        help="PyAudio input device index. Use -1 to let PyAudio choose the default input device.",
    )
    parser.add_argument("--respeaker_frames_per_buffer", type=int, default=1024)
    parser.add_argument(
        "--record_queue_max_chunks",
        type=int,
        default=8,
        help="Completed 4s capture chunks to buffer between recorder and processing; 0 means unbounded.",
    )
    parser.add_argument(
        "--respeaker_mic_channels",
        type=parse_channel_indices,
        default=parse_channel_indices("1,2,3,4"),
        help="Comma-separated zero-based ReSpeaker channels to feed DSENet.",
    )
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
        "--live_event_jsonl",
        type=Path,
        default=None,
        help="Per-run live ASR+DoA JSONL path. Defaults to <out_dir>/live_asr_doa.jsonl.",
    )
    parser.add_argument(
        "--latest_live_event_jsonl",
        type=Path,
        default=None,
        help="Stable ASR+DoA JSONL path for display scripts. Defaults to <non_dominant_results_root>/live_asr_doa_latest.jsonl.",
    )
    parser.add_argument(
        "--live_pretty_print",
        action="store_true",
        help="Print each joined ASR+DoA event from the pipeline terminal as it is emitted.",
    )
    parser.add_argument(
        "--realtime_tolerance_sec",
        type=float,
        default=0.5,
        help="Allowed cumulative lag before marking realtime_ok=0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_audio_devices:
        list_audio_devices()
        return

    if args.audio_source == "pyaudio" and args.sample_rate != args.respeaker_rate:
        raise ValueError(
            f"--sample_rate ({args.sample_rate}) must match --respeaker_rate ({args.respeaker_rate}) for live capture."
        )
    if args.raw_input_gain <= 0:
        raise ValueError("--raw_input_gain must be positive.")
    if args.audio_source == "tcp" and args.tcp_channels != 4:
        raise ValueError("This DSENet pipeline expects 4-channel TCP input. Keep --tcp_channels 4.")
    if args.streaming_mode == "managed" and not args.streaming_model_path.is_file():
        raise FileNotFoundError(f"Streaming Whisper model not found: {args.streaming_model_path}")

    args.target_doa = int(args.target_doa) % 360
    args.non_dominant_results_root.mkdir(parents=True, exist_ok=True)
    if args.out_dir is None:
        args.out_dir = next_experiment_dir(args.non_dominant_results_root)
    if args.live_event_jsonl is None:
        args.live_event_jsonl = args.out_dir / "live_asr_doa.jsonl"
    if args.latest_live_event_jsonl is None:
        args.latest_live_event_jsonl = args.non_dominant_results_root / "live_asr_doa_latest.jsonl"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
    raw_chunk_dir = args.out_dir / "raw"
    if args.save_raw_chunks:
        raw_chunk_dir.mkdir(parents=True, exist_ok=True)
    raw_concat_path = raw_chunk_dir / "raw_all_chunks.wav"
    enhanced_concat_path = enhanced_dir / "enhanced_all_chunks.wav"
    raw_concat_writer = StreamingWavWriter(raw_concat_path, args.sample_rate) if args.save_raw_chunks else None
    enhanced_concat_writer = StreamingWavWriter(enhanced_concat_path, args.sample_rate) if args.save_enhanced else None
    live_event_writer = LiveEventWriter(
        [args.live_event_jsonl, args.latest_live_event_jsonl],
        pretty_print=args.live_pretty_print,
    )

    print(f"Device: {args.device}")
    print(f"Non-dominant experiment directory: {args.out_dir}")
    print(f"Live ASR+DoA JSONL: {args.live_event_jsonl}")
    print(f"Latest ASR+DoA JSONL: {args.latest_live_event_jsonl}")
    print(f"Fixed DSENet target DoA: {args.target_doa} deg")
    print(f"Audio source: {args.audio_source}")
    print(f"Raw input gain: {args.raw_input_gain:g}x")
    if args.audio_source == "pyaudio":
        print(
            "ReSpeaker input: "
            f"index={args.respeaker_index if args.respeaker_index >= 0 else 'default'}, "
            f"rate={args.respeaker_rate}, channels={args.respeaker_channels}, "
            f"mic_channels={','.join(str(ch) for ch in args.respeaker_mic_channels)}, "
            f"chunk_seconds={args.chunk_seconds}"
        )
    else:
        print(
            "TCP audio input: "
            f"{args.tcp_host}:{args.tcp_port}, rate={args.sample_rate}, "
            f"channels={args.tcp_channels}, chunk_seconds={args.chunk_seconds}"
        )
    print(f"Streaming Whisper: {args.streaming_mode} {args.streaming_host}:{args.streaming_port}")
    print(f"Streaming Whisper model path: {args.streaming_model_path}")
    print(f"Streaming realtime sender: {args.stream_realtime}")
    print("DSENet batch size: 1")
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)

    server_proc: Optional[subprocess.Popen] = None
    if args.streaming_mode == "managed":
        print("Starting SimulStreaming Whisper server once...")
        server_proc = start_streaming_whisper_server(args)

    chunk_limit = args.max_chunks if args.max_chunks > 0 else args.max_items
    print(f"Live chunk limit: {chunk_limit if chunk_limit > 0 else 'until Ctrl+C'}")

    scene_results: List[SceneTiming] = []
    skipped_no_enhanced = 0
    timing_started = False
    emitted_transcript_keys: set[Tuple[Any, ...]] = set()

    def emit_live_events(transcripts: Sequence[Dict[str, Any]]) -> None:
        for transcript in transcripts:
            text = str(transcript.get("text") or transcript.get("raw") or "").strip()
            if not text:
                continue
            transcript_key = (
                transcript.get("start"),
                transcript.get("end"),
                text,
                transcript.get("received_wall_sec"),
            )
            if transcript_key in emitted_transcript_keys:
                continue
            emitted_transcript_keys.add(transcript_key)
            row = find_scene_row_for_transcript(scene_results, transcript)
            if row is None:
                continue
            live_event_writer.emit(live_event_from_transcript(transcript, row))

    def send_asr_fallback_row(
        *,
        fileid: int,
        chunk_index: int,
        mic_name: str,
        wav_tc: np.ndarray,
        sr: int,
        duration_sec: float,
        pred_doas: Sequence[int],
        fixed_doa_sec: float,
        dse_sec: float,
        reason: str,
    ) -> None:
        if args.skip_asr_policy == "drop":
            return

        fallback_audio = fallback_audio_for_skipped_chunk(wav_tc, args.skip_asr_policy)
        audio_start_sec = stream_client.total_audio_sec
        transcript_start_idx = stream_client.transcript_count()
        stream_paced_send_sec, chunk_audio_sec = stream_client.send_audio(fallback_audio, sr)
        transcript_delta = stream_client.transcripts_since(transcript_start_idx)
        received_during_send_text = transcript_text(transcript_delta)

        frontend_compute_sec = fixed_doa_sec + dse_sec
        audio_end_sec = audio_start_sec + chunk_audio_sec
        pipeline_wall_sec = time.perf_counter() - stream_start
        stream_audio_end_sec = stream_client.total_audio_sec
        pipeline_lag_sec = pipeline_wall_sec - stream_audio_end_sec
        realtime_ok = int(pipeline_lag_sec <= args.realtime_tolerance_sec)
        selected_save_name = f"{reason}_{args.skip_asr_policy}_chunk{chunk_index}.wav"
        if args.save_enhanced:
            sf.write(str(enhanced_dir / selected_save_name), fallback_audio, sr)
            if enhanced_concat_writer is not None:
                enhanced_concat_writer.write(fallback_audio)
        row = SceneTiming(
            fileid=fileid,
            chunk_index=chunk_index,
            mic_file=mic_name,
            duration_sec=duration_sec,
            audio_start_sec=audio_start_sec,
            audio_end_sec=audio_end_sec,
            predicted_doa_count=len(pred_doas),
            predicted_doas=",".join(str(doa) for doa in pred_doas),
            asr_audio_kind=f"fallback_{args.skip_asr_policy}_{reason}",
            selected_enhanced_index=-1,
            selected_doa=-1,
            selected_rms=signal_rms(fallback_audio),
            selection_policy="fixed_doa",
            selection_scores="",
            selection_active_ratio=0.0,
            selected_enhanced_file=selected_save_name,
            fixed_doa_sec=fixed_doa_sec,
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
            fixed_doa_rtf=fixed_doa_sec / duration_sec,
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
        scene_results.append(row)
        emit_live_events(transcript_delta)

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
        if raw_concat_writer is not None:
            raw_concat_writer.close()
        if enhanced_concat_writer is not None:
            enhanced_concat_writer.close()
        live_event_writer.close()
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        raise

    try:
        input_device_index = None if args.respeaker_index < 0 else args.respeaker_index
        if args.audio_source == "pyaudio":
            source_context = ReSpeakerChunkSource(
                sample_rate=args.respeaker_rate,
                input_channels=args.respeaker_channels,
                sample_width=args.respeaker_width,
                input_device_index=input_device_index,
                frames_per_buffer=args.respeaker_frames_per_buffer,
                mic_channels=args.respeaker_mic_channels,
                chunk_seconds=args.chunk_seconds,
                queue_max_chunks=args.record_queue_max_chunks,
            )
            source_message = "Recording from ReSpeaker in a background thread. Press Ctrl+C to stop."
        else:
            source_context = TcpInt16ChunkSource(
                host=args.tcp_host,
                port=args.tcp_port,
                sample_rate=args.sample_rate,
                channels=args.tcp_channels,
                chunk_seconds=args.chunk_seconds,
            )
            source_message = "Receiving ReSpeaker audio over TCP. Press Ctrl+C to stop."

        with source_context as chunk_source:
            print(source_message)
            live_chunks = enumerate(chunk_source.iter_chunks(max_chunks=chunk_limit), start=1)
            for chunk_index, wav_tc in tqdm(
                live_chunks,
                total=chunk_limit if chunk_limit > 0 else None,
                desc="Live ReSpeaker",
                unit="chunk",
            ):
                if not timing_started:
                    stream_start = time.perf_counter()
                    stream_client.start_time = stream_start
                    timing_started = True
                    print("Realtime clock started at first received audio chunk.")

                fileid = 1
                sr = args.sample_rate
                mic_name = f"respeaker_live_chunk_{chunk_index:06d}.wav"
                if args.raw_input_gain != 1.0:
                    wav_tc = np.clip(wav_tc * args.raw_input_gain, -1.0, 1.0).astype(np.float32)
                if args.save_raw_chunks:
                    sf.write(str(raw_chunk_dir / mic_name), wav_tc, sr)
                    if raw_concat_writer is not None:
                        raw_concat_writer.write(wav_tc)
                duration_sec = wav_tc.shape[0] / float(sr)
                noisy_ct = torch.from_numpy(wav_tc.T.copy())
                fixed_doa_sec = 0.0
                pred_doas = [args.target_doa]

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
                    skipped_no_enhanced += 1
                    print(
                        f"live chunk={chunk_index}: DSENet produced no enhanced output, "
                        f"using {args.skip_asr_policy} fallback."
                    )
                    send_asr_fallback_row(
                        fileid=fileid,
                        chunk_index=chunk_index,
                        mic_name=mic_name,
                        wav_tc=wav_tc,
                        sr=sr,
                        duration_sec=duration_sec,
                        pred_doas=pred_doas,
                        fixed_doa_sec=fixed_doa_sec,
                        dse_sec=dse_sec,
                        reason="skip_no_enhanced",
                    )
                    continue

                selected_idx = 0
                selected_doa = args.target_doa
                enhanced_for_asr = enhanced_batch[selected_idx]
                selected_rms = signal_rms(enhanced_for_asr)
                selection_scores = [selected_rms]
                selection_active_ratio = 1.0
                selected_save_name = (
                    f"enhanced_live_chunk{chunk_index:06d}_"
                    f"fixed_doa{selected_doa:03d}.wav"
                )

                if args.save_enhanced:
                    sf.write(str(enhanced_dir / selected_save_name), enhanced_for_asr, sr)
                    if enhanced_concat_writer is not None:
                        enhanced_concat_writer.write(enhanced_for_asr)

                audio_start_sec = stream_client.total_audio_sec
                transcript_start_idx = stream_client.transcript_count()
                stream_paced_send_sec, chunk_audio_sec = stream_client.send_audio(enhanced_for_asr, sr)
                transcript_delta = stream_client.transcripts_since(transcript_start_idx)
                received_during_send_text = transcript_text(transcript_delta)

                frontend_compute_sec = fixed_doa_sec + dse_sec
                audio_end_sec = audio_start_sec + chunk_audio_sec
                pipeline_wall_sec = time.perf_counter() - stream_start
                stream_audio_end_sec = stream_client.total_audio_sec
                pipeline_lag_sec = pipeline_wall_sec - stream_audio_end_sec
                realtime_ok = int(pipeline_lag_sec <= args.realtime_tolerance_sec)
                row = SceneTiming(
                    fileid=fileid,
                    chunk_index=chunk_index,
                    mic_file=mic_name,
                    duration_sec=duration_sec,
                    audio_start_sec=audio_start_sec,
                    audio_end_sec=audio_end_sec,
                    predicted_doa_count=len(pred_doas),
                    predicted_doas=",".join(str(doa) for doa in pred_doas),
                    asr_audio_kind="enhanced",
                    selected_enhanced_index=selected_idx,
                    selected_doa=selected_doa,
                    selected_rms=selected_rms,
                    selection_policy="fixed_doa",
                    selection_scores=",".join(f"{score:.8g}" for score in selection_scores),
                    selection_active_ratio=selection_active_ratio,
                    selected_enhanced_file=selected_save_name,
                    fixed_doa_sec=fixed_doa_sec,
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
                    fixed_doa_rtf=fixed_doa_sec / duration_sec,
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
                scene_results.append(row)
                emit_live_events(transcript_delta)
    except KeyboardInterrupt:
        print("\nStopping live ReSpeaker capture and writing results collected so far...")
    finally:
        stream_client.close(args.stream_final_wait)
        if raw_concat_writer is not None:
            raw_concat_writer.close()
        if enhanced_concat_writer is not None:
            enhanced_concat_writer.close()
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    all_transcripts = stream_client.all_transcripts()
    emit_live_events(all_transcripts)
    live_event_writer.close()
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

    chunk_transcript_results: List[ChunkTranscript] = []
    for row in sorted(scene_results, key=lambda r: (r.fileid, r.chunk_index)):
        chunk_transcript_results.append(
            ChunkTranscript(
                chunk_index=row.chunk_index,
                mic_file=row.mic_file,
                audio_start_sec=row.audio_start_sec,
                audio_end_sec=row.audio_end_sec,
                duration_sec=row.duration_sec,
                asr_audio_kind=row.asr_audio_kind,
                predicted_doa_count=row.predicted_doa_count,
                predicted_doas=row.predicted_doas,
                selected_doa=row.selected_doa,
                pipeline_lag_sec=row.pipeline_lag_sec,
                timestamp_assigned_text=row.timestamp_assigned_text,
                hypothesis_text_cleaned=clean_asr_hypothesis_text(row.timestamp_assigned_text),
            )
        )

    final_audio_sec = stream_client.total_audio_sec
    final_wall_sec = time.perf_counter() - stream_start if timing_started else 0.0
    final_lag_sec = final_wall_sec - final_audio_sec
    final_cumulative_rtf = final_wall_sec / final_audio_sec if final_audio_sec > 0 else 0.0

    details_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_details_1asr_4s_full.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    chunk_transcripts_csv = args.out_dir / f"pipeline_streaming_{args.whisper_model}_chunk_transcripts_4s_full.csv"
    with chunk_transcripts_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(ChunkTranscript.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in chunk_transcript_results:
            writer.writerow(asdict(row))

    summary = {
        "audio_source": args.audio_source,
        "out_dir": str(args.out_dir),
        "non_dominant_results_root": str(args.non_dominant_results_root),
        "live_event_jsonl": str(args.live_event_jsonl),
        "latest_live_event_jsonl": str(args.latest_live_event_jsonl),
        "raw_dir": str(raw_chunk_dir) if args.save_raw_chunks else "",
        "enhanced_dir": str(enhanced_dir) if args.save_enhanced else "",
        "raw_concat_wav": str(raw_concat_path) if args.save_raw_chunks else "",
        "enhanced_concat_wav": str(enhanced_concat_path) if args.save_enhanced else "",
        "tcp_host": args.tcp_host,
        "tcp_port": args.tcp_port,
        "tcp_channels": args.tcp_channels,
        "respeaker_rate": args.respeaker_rate,
        "respeaker_channels": args.respeaker_channels,
        "respeaker_width": args.respeaker_width,
        "respeaker_index": args.respeaker_index,
        "respeaker_frames_per_buffer": args.respeaker_frames_per_buffer,
        "record_queue_max_chunks": args.record_queue_max_chunks,
        "respeaker_mic_channels": ",".join(str(ch) for ch in args.respeaker_mic_channels),
        "chunk_seconds": args.chunk_seconds,
        "raw_input_gain": args.raw_input_gain,
        "timing_origin": "first_received_audio_chunk",
        "max_chunks": chunk_limit,
        "save_raw_chunks": args.save_raw_chunks,
        "save_enhanced": args.save_enhanced,
        "dse_ckpt": str(args.dse_ckpt),
        "fixed_target_doa": args.target_doa,
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
        "dse_batch_size": 1,
        "selection": "fixed_doa",
        "skip_asr_policy": args.skip_asr_policy,
        "evaluated_chunks": len(scene_results),
        "chunk_transcript_rows": len(chunk_transcript_results),
        "skipped_no_enhanced": skipped_no_enhanced,
        "mean_duration_sec": float(np.mean([r.duration_sec for r in scene_results])) if scene_results else 0.0,
        "mean_fixed_doa_sec": float(np.mean([r.fixed_doa_sec for r in scene_results])) if scene_results else 0.0,
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
    print(f"Chunk transcript rows: {summary['chunk_transcript_rows']}")
    print(f"Fixed target DoA: {summary['fixed_target_doa']} deg")
    print(f"Final audio sent: {summary['final_audio_sec']:.3f}s")
    print(f"Final pipeline wall time: {summary['final_pipeline_wall_sec']:.3f}s")
    print(f"Final pipeline lag: {summary['final_pipeline_lag_sec']:.3f}s")
    print(f"Final cumulative RTF: {summary['final_cumulative_rtf']:.3f}")
    print(f"Realtime tolerance: {summary['realtime_tolerance_sec']:.3f}s")
    print(f"Final realtime ok: {summary['final_realtime_ok']}")
    print(
        "Mean timing per scene: "
        f"fixed_doa={summary['mean_fixed_doa_sec']:.3f}s, "
        f"DSENet={summary['mean_dsenet_sec']:.3f}s, "
        f"frontend_compute={summary['mean_frontend_compute_sec']:.3f}s, "
        f"paced_send={summary['mean_stream_paced_send_sec']:.3f}s"
    )
    print(f"Transcript segments received: {summary['stream_transcript_segments']}")
    print(f"Transcript segments timestamp-assigned: {summary['timestamp_assigned_transcript_segments']}")
    print(f"Non-dominant experiment directory: {args.out_dir}")
    print(f"Saved live ASR+DoA JSONL: {args.live_event_jsonl}")
    print(f"Saved latest ASR+DoA JSONL: {args.latest_live_event_jsonl}")
    if args.save_raw_chunks:
        print(f"Saved raw chunks: {raw_chunk_dir}")
        print(f"Saved concatenated raw audio: {raw_concat_path}")
    if args.save_enhanced:
        print(f"Saved fixed-DoA enhanced chunks: {enhanced_dir}")
        print(f"Saved concatenated fixed-DoA enhanced audio: {enhanced_concat_path}")
    print(f"Saved streaming details: {details_csv}")
    print(f"Saved chunk transcripts: {chunk_transcripts_csv}")
    print(f"Saved transcripts: {transcript_jsonl}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
