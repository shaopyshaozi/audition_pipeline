"""
WSL helper for checking and recording ReSpeaker-style multichannel audio.

This script is intentionally small and direct:
1. Show what WSL/PyAudio can see.
2. Record 6-channel int16 audio when the ReSpeaker is actually visible.
3. Save channels 1,2,3,4 as a 4-channel wav for IPDNET/DSENet checks.

If WSL only lists "pulse" and "default", the ReSpeaker USB mic has not been
passed through as a real ALSA input device. In that case, use Windows Python or
attach the USB device to WSL before running the full realtime pipeline.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np


DEFAULT_RATE = 16000
DEFAULT_INPUT_CHANNELS = 6
DEFAULT_OUTPUT_CHANNELS = "1,2,3,4"
DEFAULT_WIDTH_BYTES = 2
DEFAULT_FRAMES_PER_BUFFER = 1024
DEFAULT_CHUNK_SECONDS = 4.0
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("output_raw_4ch.wav")
DEFAULT_RECORDINGS_DIR = Path(__file__).resolve().with_name("recordings")


def parse_channel_indices(value: str) -> List[int]:
    channels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not channels:
        raise argparse.ArgumentTypeError("At least one output channel is required.")
    if len(channels) != len(set(channels)):
        raise argparse.ArgumentTypeError(f"Duplicate channel index in: {value}")
    return channels


def is_wsl() -> bool:
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        version = ""
    return "microsoft" in version or "wsl" in version


def run_probe_command(command: List[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("  command not found")
        return
    output = result.stdout.strip()
    print(output if output else "  no output")


def print_wsl_probe() -> None:
    print(f"Platform: {platform.platform()}")
    print(f"Running in WSL: {is_wsl()}")
    run_probe_command(["arecord", "-l"])
    run_probe_command(["arecord", "-L"])
    run_probe_command(["pactl", "list", "sources", "short"])


def import_pyaudio():
    try:
        import pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "PyAudio is not installed in this Python environment. "
            "Install it in the venv first, then rerun this script."
        ) from exc
    return pyaudio


def list_pyaudio_devices() -> List[dict]:
    pyaudio = import_pyaudio()
    pa = pyaudio.PyAudio()
    devices: List[dict] = []
    try:
        print("\nAvailable PyAudio input devices:")
        for idx in range(pa.get_device_count()):
            info = dict(pa.get_device_info_by_index(idx))
            if int(info.get("maxInputChannels", 0)) <= 0:
                continue
            devices.append(info)
            name = str(info.get("name", ""))
            channels = int(info.get("maxInputChannels", 0))
            rate = float(info.get("defaultSampleRate", 0.0))
            print(f"  [{idx}] {name} | input_channels={channels} | default_rate={rate:g}")
    finally:
        pa.terminate()

    if not devices:
        print("  no PyAudio input devices found")
    return devices


def looks_like_virtual_wsl_audio(device_name: str) -> bool:
    normalized = device_name.lower().strip()
    return normalized in {"pulse", "default"} or "pulse" in normalized


def choose_device_index(devices: List[dict], requested_index: Optional[int], allow_virtual: bool) -> Optional[int]:
    if requested_index is not None:
        return requested_index

    real_candidates = [
        int(device["index"])
        for device in devices
        if not looks_like_virtual_wsl_audio(str(device.get("name", "")))
    ]
    if real_candidates:
        return real_candidates[0]

    if allow_virtual and devices:
        return int(devices[0]["index"])

    return None


def resample_int16_chunk(
    audio: np.ndarray,
    input_rate: int,
    output_rate: int,
    output_samples: int,
) -> np.ndarray:
    if input_rate == output_rate and audio.shape[0] == output_samples:
        return audio.astype(np.int16, copy=False)

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(input_rate, output_rate)
        up = output_rate // gcd
        down = input_rate // gcd
        resampled = resample_poly(audio.astype(np.float32), up, down, axis=0)
    except ImportError:
        old_x = np.arange(audio.shape[0], dtype=np.float64)
        new_x = np.linspace(0, max(audio.shape[0] - 1, 0), output_samples, dtype=np.float64)
        resampled = np.stack(
            [np.interp(new_x, old_x, audio[:, ch]).astype(np.float32) for ch in range(audio.shape[1])],
            axis=1,
        )

    if resampled.shape[0] != output_samples:
        if resampled.shape[0] > output_samples:
            resampled = resampled[:output_samples]
        else:
            pad = np.repeat(resampled[-1:, :], output_samples - resampled.shape[0], axis=0)
            resampled = np.concatenate((resampled, pad), axis=0)
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def record_audio(args: argparse.Namespace) -> None:
    pyaudio = import_pyaudio()
    output_channels = args.output_channels
    invalid_channels = [ch for ch in output_channels if ch < 0 or ch >= args.input_channels]
    if invalid_channels:
        raise ValueError(
            f"Output channels {invalid_channels} are outside input channel range 0..{args.input_channels - 1}."
        )

    devices = list_pyaudio_devices()
    device_index = choose_device_index(devices, args.device_index, args.allow_virtual)
    if device_index is None:
        raise RuntimeError(
            "No real multichannel input device was found in WSL. "
            "If you only see 'pulse'/'default', attach the ReSpeaker USB device to WSL "
            "or run this recorder from Windows Python. Use --allow_virtual only for debugging."
        )

    selected = next((item for item in devices if int(item["index"]) == device_index), None)
    if selected is not None:
        name = str(selected.get("name", ""))
        max_channels = int(selected.get("maxInputChannels", 0))
        if looks_like_virtual_wsl_audio(name) and not args.allow_virtual:
            raise RuntimeError(
                f"Device [{device_index}] is '{name}', which is WSL virtual audio, not the raw ReSpeaker. "
                "Pass the USB mic through to WSL or run from Windows Python."
            )
        if max_channels < args.input_channels:
            raise RuntimeError(
                f"Device [{device_index}] only reports {max_channels} input channels, "
                f"but --input_channels is {args.input_channels}."
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames: List[bytes] = []
    stop_event = threading.Event()

    pa = pyaudio.PyAudio()
    stream = None
    try:
        stream = pa.open(
            rate=args.rate,
            format=pa.get_format_from_width(args.width_bytes),
            channels=args.input_channels,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=args.frames_per_buffer,
        )

        def read_loop() -> None:
            print("* recording... Press Enter to stop.")
            while not stop_event.is_set():
                data = stream.read(args.frames_per_buffer, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).reshape(-1, args.input_channels)
                raw_selected = audio[:, output_channels]
                frames.append(raw_selected.astype(np.int16).tobytes())

        input("Press Enter to start recording...")
        thread = threading.Thread(target=read_loop, daemon=True)
        thread.start()

        if args.seconds > 0:
            thread.join(timeout=args.seconds)
            stop_event.set()
        else:
            input()
            stop_event.set()
        thread.join()

    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()

    audio_all = np.frombuffer(b"".join(frames), dtype=np.int16).reshape(-1, len(output_channels))

    if args.seconds > 0:
        target_samples = int(round(args.seconds * args.rate))
        audio_all = audio_all[:target_samples]

    with wave.open(str(args.output), "wb") as wf:
        wf.setnchannels(len(output_channels))
        wf.setsampwidth(args.width_bytes)
        wf.setframerate(args.rate)
        wf.writeframes(audio_all.astype(np.int16).tobytes())

    duration = 0.0
    if frames:
        samples = sum(len(frame) for frame in frames) / (args.width_bytes * len(output_channels))
        duration = samples / float(args.rate)
    print(f"* done recording: {duration:.2f}s")
    print(f"Saved {len(output_channels)}ch wav: {args.output}")


def send_tcp_audio(args: argparse.Namespace) -> None:
    pyaudio = import_pyaudio()
    capture_rate = args.capture_rate if args.capture_rate is not None else args.rate
    output_channels = args.output_channels
    invalid_channels = [ch for ch in output_channels if ch < 0 or ch >= args.input_channels]
    if invalid_channels:
        raise ValueError(
            f"Output channels {invalid_channels} are outside input channel range 0..{args.input_channels - 1}."
        )
    if len(output_channels) != 4:
        raise ValueError("The WSL IPDNET/DSENet receiver expects exactly 4 output channels.")

    devices = list_pyaudio_devices()
    device_index = choose_device_index(devices, args.device_index, args.allow_virtual)
    if device_index is None:
        raise RuntimeError("No input device found. Use --list to find the Windows ReSpeaker index.")

    selected = next((item for item in devices if int(item["index"]) == device_index), None)
    if selected is not None:
        max_channels = int(selected.get("maxInputChannels", 0))
        if max_channels < args.input_channels:
            raise RuntimeError(
                f"Device [{device_index}] only reports {max_channels} input channels, "
                f"but --input_channels is {args.input_channels}."
            )

    frames_sent = 0
    chunks_sent = 0
    chunk_samples = int(round(args.rate * args.chunk_seconds))
    capture_chunk_samples = int(round(capture_rate * args.chunk_seconds))
    if chunk_samples <= 0:
        raise ValueError("--chunk_seconds must be positive.")
    if capture_chunk_samples <= 0:
        raise ValueError("--chunk_seconds and --capture_rate must produce a positive chunk size.")
    if args.save_chunks:
        args.recordings_dir.mkdir(parents=True, exist_ok=True)

    pa = pyaudio.PyAudio()
    stream = None
    sock = None
    try:
        print(f"\nConnecting to WSL receiver at {args.tcp_host}:{args.tcp_port}...")
        sock = socket.create_connection((args.tcp_host, args.tcp_port), timeout=30.0)
        sock.settimeout(None)
        print("Connected. Starting ReSpeaker capture.")
        print(f"Capture rate: {capture_rate} Hz; transmit/save rate: {args.rate} Hz")

        stream = pa.open(
            rate=capture_rate,
            format=pa.get_format_from_width(args.width_bytes),
            channels=args.input_channels,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=args.frames_per_buffer,
        )

        print(f"* sending {args.chunk_seconds:.3f}s chunks. Press Ctrl+C to stop.")
        pending = np.empty((0, len(output_channels)), dtype=np.int16)
        while True:
            while pending.shape[0] < capture_chunk_samples:
                data = stream.read(args.frames_per_buffer, exception_on_overflow=False)
                audio = np.frombuffer(data, dtype=np.int16).reshape(-1, args.input_channels)
                raw_selected = audio[:, output_channels]
                pending = np.concatenate((pending, raw_selected.astype(np.int16, copy=False)), axis=0)

            capture_chunk = pending[:capture_chunk_samples]
            pending = pending[capture_chunk_samples:]
            chunk = resample_int16_chunk(
                capture_chunk,
                input_rate=capture_rate,
                output_rate=args.rate,
                output_samples=chunk_samples,
            )
            sock.sendall(chunk.astype("<i2", copy=False).tobytes())
            frames_sent += chunk.shape[0]
            chunks_sent += 1
            if args.save_chunks:
                chunk_path = args.recordings_dir / f"respeaker_4ch_chunk_{chunks_sent:06d}.wav"
                with wave.open(str(chunk_path), "wb") as wf:
                    wf.setnchannels(len(output_channels))
                    wf.setsampwidth(args.width_bytes)
                    wf.setframerate(args.rate)
                    wf.writeframes(chunk.astype(np.int16, copy=False).tobytes())
            print(f"  sent chunk {chunks_sent}: {chunk.shape[0] / float(args.rate):.3f}s")
    except KeyboardInterrupt:
        print("\nStopping TCP sender...")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        pa.terminate()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            sock.close()

    duration = frames_sent / float(args.rate)
    print(f"* done sending: {duration:.2f}s in {chunks_sent} chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSL ReSpeaker device probe and 4-channel recorder.")
    parser.add_argument("--list", action="store_true", help="List PyAudio devices and exit.")
    parser.add_argument("--probe", action="store_true", help="Print WSL ALSA/PulseAudio probe information and exit.")
    parser.add_argument("--send_tcp", action="store_true", help="Send selected ReSpeaker channels to the WSL TCP receiver.")
    parser.add_argument("--tcp_host", type=str, default="127.0.0.1", help="WSL receiver host/IP for --send_tcp.")
    parser.add_argument("--tcp_port", type=int, default=50007, help="WSL receiver port for --send_tcp.")
    parser.add_argument("--device_index", type=int, default=None, help="PyAudio input device index.")
    parser.add_argument("--allow_virtual", action="store_true", help="Allow WSL pulse/default device for debugging.")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE, help="Transmit/save sample rate.")
    parser.add_argument(
        "--capture_rate",
        type=int,
        default=None,
        help="Input device capture rate. Defaults to --rate; use 44100 for devices that cannot open at 16000.",
    )
    parser.add_argument("--input_channels", type=int, default=DEFAULT_INPUT_CHANNELS)
    parser.add_argument("--output_channels", type=parse_channel_indices, default=parse_channel_indices(DEFAULT_OUTPUT_CHANNELS))
    parser.add_argument("--width_bytes", type=int, default=DEFAULT_WIDTH_BYTES)
    parser.add_argument("--frames_per_buffer", type=int, default=DEFAULT_FRAMES_PER_BUFFER)
    parser.add_argument("--chunk_seconds", type=float, default=DEFAULT_CHUNK_SECONDS, help="TCP sender chunk duration.")
    parser.add_argument("--save_chunks", action="store_true", help="Save each outgoing TCP chunk as a local wav.")
    parser.add_argument("--recordings_dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--seconds", type=float, default=0.0, help="Fixed local recording length. 0 means press Enter to stop.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe:
        print_wsl_probe()
    if args.list:
        list_pyaudio_devices()
    if args.probe or args.list:
        return
    if args.send_tcp:
        send_tcp_audio(args)
        return
    record_audio(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
