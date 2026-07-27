"""
Post-process enhanced speech WAVs from the Respeaker pipeline.

Default chain:
1. Split into 4 s chunks to mimic the streaming pipeline.
2. High-pass filter each chunk to remove low-frequency rumble.
3. Aggressively suppress quiet/stationary noise in each chunk.
4. Restore each processed chunk to its original RMS by default.
5. Optionally apply global RMS normalization after concatenation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, istft, stft


def db_to_amp(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db) / 20.0)


def amp_to_db(amp: np.ndarray | float, floor_db: float = -120.0) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(amp, db_to_amp(floor_db)))


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return audio
    nyquist = sr / 2.0
    cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(4, cutoff, btype="highpass")
    return filtfilt(b, a, audio).astype(np.float32)


def spectral_denoise(
    audio: np.ndarray,
    sr: int,
    noise_reduce_db: float,
    noise_percentile: float,
    noise_profile_percentile: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    if noise_reduce_db <= 0:
        return audio

    _, _, spec = stft(
        audio,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        boundary="zeros",
    )
    mag = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))
    frame_energy = np.mean(mag, axis=0)
    threshold = np.percentile(frame_energy, noise_percentile)
    noise_frames = mag[:, frame_energy <= threshold]
    if noise_frames.size == 0:
        noise_frames = mag

    noise_mag = np.percentile(
        noise_frames,
        np.clip(noise_profile_percentile, 0.0, 100.0),
        axis=1,
        keepdims=True,
    )
    floor = db_to_amp(-noise_reduce_db)
    gain = 1.0 - noise_mag / np.maximum(mag, 1e-8)
    gain = np.clip(gain, floor, 1.0)
    cleaned_spec = mag * gain * phase
    _, cleaned = istft(
        cleaned_spec,
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        input_onesided=True,
    )
    return cleaned[: len(audio)].astype(np.float32)


def soft_knee_compressor(
    audio: np.ndarray,
    threshold_db: float,
    ratio: float,
    makeup_db: float,
) -> np.ndarray:
    if ratio <= 1.0:
        return audio * db_to_amp(makeup_db)

    sign = np.sign(audio)
    mag = np.abs(audio)
    level_db = amp_to_db(mag)
    over_db = np.maximum(level_db - threshold_db, 0.0)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
    gain_db = makeup_db - gain_reduction_db
    return (sign * mag * db_to_amp(gain_db)).astype(np.float32)


def peak_limit(audio: np.ndarray, limit: float) -> np.ndarray:
    limit = float(np.clip(limit, 0.01, 1.0))
    return np.clip(audio, -limit, limit).astype(np.float32)


def signal_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12))


def normalize_rms(audio: np.ndarray, target_rms_dbfs: float, peak_limit_value: float) -> np.ndarray:
    rms = signal_rms(audio)
    if rms <= 0:
        return audio
    target_rms = db_to_amp(target_rms_dbfs)
    audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    if peak > peak_limit_value:
        audio = audio * (peak_limit_value / peak)
    return audio.astype(np.float32)


def match_rms(audio: np.ndarray, reference_audio: np.ndarray, peak_limit_value: float, blend: float) -> np.ndarray:
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0:
        return audio.astype(np.float32)

    audio_rms = signal_rms(audio)
    reference_rms = signal_rms(reference_audio)
    if audio_rms <= 0 or reference_rms <= 0:
        return audio.astype(np.float32)

    gain = (reference_rms / audio_rms) ** blend
    audio = audio * gain
    return peak_limit(audio, peak_limit_value)


def postprocess_audio(audio: np.ndarray, sr: int, args: argparse.Namespace) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    original_audio = audio.copy()
    audio = highpass(audio, sr, args.highpass_hz)
    audio = spectral_denoise(
        audio,
        sr,
        noise_reduce_db=args.noise_reduce_db,
        noise_percentile=args.noise_percentile,
        noise_profile_percentile=args.noise_profile_percentile,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    audio = soft_knee_compressor(
        audio,
        threshold_db=args.compressor_threshold_db,
        ratio=args.compressor_ratio,
        makeup_db=args.makeup_db,
    )
    audio = peak_limit(audio, args.peak_limit)
    if args.preserve_chunk_rms:
        audio = match_rms(audio, original_audio, args.peak_limit, args.preserve_chunk_rms_blend)
    elif args.target_rms_dbfs is not None:
        audio = normalize_rms(audio, args.target_rms_dbfs, args.peak_limit)
    return peak_limit(audio, args.peak_limit)


def postprocess_audio_in_chunks(audio: np.ndarray, sr: int, args: argparse.Namespace) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)

    if not args.process_in_chunks or args.chunk_sec <= 0:
        return postprocess_audio(audio, sr, args)

    chunk_samples = max(1, int(round(args.chunk_sec * sr)))
    processed_chunks = []
    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start:start + chunk_samples]
        if len(chunk) == 0:
            continue
        processed_chunks.append(postprocess_audio(chunk, sr, args))
    if not processed_chunks:
        return audio
    processed = np.concatenate(processed_chunks).astype(np.float32)
    if args.global_target_rms_dbfs is not None:
        processed = normalize_rms(processed, args.global_target_rms_dbfs, args.peak_limit)
    return peak_limit(processed, args.peak_limit)


def output_path_for(input_path: Path, input_root: Path, output_dir: Path, suffix: str) -> Path:
    if input_path == input_root:
        return output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
    rel = input_path.relative_to(input_root)
    return output_dir / rel.with_name(f"{rel.stem}{suffix}{rel.suffix}")


def iter_wavs(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    yield from sorted(input_path.rglob("*.wav"))


def process_file(input_path: Path, input_root: Path, output_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    audio, sr = sf.read(str(input_path), always_2d=False)
    processed = postprocess_audio_in_chunks(audio, sr, args)
    out_path = output_path_for(input_path, input_root, output_dir, args.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), processed, sr)
    if args.save_chunks and args.process_in_chunks and args.chunk_sec > 0:
        chunk_samples = max(1, int(round(args.chunk_sec * sr)))
        chunk_dir = output_dir / "chunks" / input_path.stem
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for chunk_idx, start in enumerate(range(0, len(processed), chunk_samples)):
            chunk = processed[start:start + chunk_samples]
            sf.write(str(chunk_dir / f"{input_path.stem}_chunk{chunk_idx:03d}.wav"), chunk, sr)
    return input_path, out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process enhanced speech WAVs.")
    parser.add_argument("input", type=Path, help="Input WAV file or folder containing WAVs.")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--suffix", type=str, default="_postprocessed")
    parser.add_argument("--chunk_sec", type=float, default=4.0)
    parser.add_argument(
        "--process_in_chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process audio independently in chunk_sec chunks before concatenating the result.",
    )
    parser.add_argument("--save_chunks", action="store_true", help="Also save each processed chunk separately.")
    parser.add_argument("--highpass_hz", type=float, default=100.0)
    parser.add_argument("--noise_reduce_db", type=float, default=500.0)
    parser.add_argument("--noise_percentile", type=float, default=80.0)
    parser.add_argument(
        "--noise_profile_percentile",
        type=float,
        default=85.0,
        help="Percentile used to summarize selected noise frames when noise_reduce_db is enabled.",
    )
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--compressor_threshold_db", type=float, default=-28.0)
    parser.add_argument("--compressor_ratio", type=float, default=1.0)
    parser.add_argument("--makeup_db", type=float, default=0.0)
    parser.add_argument(
        "--preserve_chunk_rms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After per-chunk cleanup, restore each chunk toward its original enhanced RMS.",
    )
    parser.add_argument(
        "--preserve_chunk_rms_blend",
        type=float,
        default=1.0,
        help="0 disables RMS restore; 1 fully restores original chunk RMS.",
    )
    parser.add_argument(
        "--target_rms_dbfs",
        type=float,
        default=None,
        help="Per-chunk RMS target used only when --no-preserve_chunk_rms is set.",
    )
    parser.add_argument(
        "--global_target_rms_dbfs",
        type=float,
        default=None,
        help="Optional RMS normalization applied once after processed chunks are concatenated.",
    )
    parser.add_argument("--peak_limit", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_path.parent / "postprocessed" if input_path.is_file() else input_path / "postprocessed"
    output_dir = output_dir.resolve()

    count = 0
    for wav_path in iter_wavs(input_path):
        in_path, out_path = process_file(wav_path.resolve(), input_path, output_dir, args)
        print(f"{in_path} -> {out_path}")
        count += 1
    print(f"Processed {count} wav file(s).")


if __name__ == "__main__":
    main()
