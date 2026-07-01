#!/usr/bin/env python3
import argparse
import socket
import time
from pathlib import Path
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SR = 16000
BYTES_PER_SAMPLE = 2  # int16
BYTES_PER_SECOND = TARGET_SR * BYTES_PER_SAMPLE


def load_wav_as_pcm16(path: Path) -> bytes:
    audio, sr = sf.read(str(path), dtype="float32")

    # mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample to 16 kHz
    if sr != TARGET_SR:
        audio = resample_poly(audio, TARGET_SR, sr)

    # avoid clipping
    audio = np.clip(audio, -1.0, 1.0)

    # float32 [-1, 1] -> int16 little endian
    pcm16 = (audio * 32767.0).astype("<i2")
    return pcm16.tobytes()


def get_fileid(path: Path):
    match = re.search(r"fileid_(\d+)", path.name)
    return int(match.group(1)) if match else 10**9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir",
        default="/mnt/d/邵鹏远/UCL/博1/code/DSENet/eval/enhanced_wer_4mics_dominant",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=43001)
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--chunk_ms",
        type=int,
        default=100,
        help="Audio packet size in milliseconds. Use 100 for microphone-like streaming.",
    )
    args = parser.parse_args()

    wav_paths = sorted(Path(args.wav_dir).glob(args.pattern), key=get_fileid)
    print(f"Found {len(wav_paths)} wav files")

    chunk_size = int(BYTES_PER_SECOND * args.chunk_ms / 1000)

    global_start = time.time()
    total_bytes_sent = 0

    with socket.create_connection((args.host, args.port)) as sock:
        for wav_path in wav_paths:
            wall_time = time.time() - global_start
            audio_t = total_bytes_sent / BYTES_PER_SECOND
            gap = wall_time - audio_t

            print(
                f"START {wav_path.name}: "
                f"wall={wall_time:.3f}s | "
                f"audio_t={audio_t:.3f}s | "
                f"gap={gap:.3f}s"
            )

            prep_start = time.time()
            pcm_bytes = load_wav_as_pcm16(wav_path)
            prep_time = time.time() - prep_start

            audio_duration = len(pcm_bytes) / BYTES_PER_SECOND

            print(
                f"duration={audio_duration:.3f}s | "
                f"prep={prep_time:.3f}s"
            )

            if args.realtime:
                for i in range(0, len(pcm_bytes), chunk_size):
                    chunk = pcm_bytes[i:i + chunk_size]
                    sock.sendall(chunk)

                    total_bytes_sent += len(chunk)

                    expected_wall_time = total_bytes_sent / BYTES_PER_SECOND
                    actual_wall_time = time.time() - global_start
                    sleep_time = expected_wall_time - actual_wall_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)
            else:
                sock.sendall(pcm_bytes)
                total_bytes_sent += len(pcm_bytes)

        final_wall = time.time() - global_start
        final_audio = total_bytes_sent / BYTES_PER_SECOND
        print(
            f"Finished sending all wav files | "
            f"wall={final_wall:.3f}s | "
            f"audio={final_audio:.3f}s | "
            f"gap={final_wall - final_audio:.3f}s"
        )


if __name__ == "__main__":
    main()