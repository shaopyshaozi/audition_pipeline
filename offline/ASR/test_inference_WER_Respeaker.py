from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import whisper
from scipy.signal import resample_poly
from tqdm import tqdm


def load_audio_mono(audio_path: Path, channel: int = 0) -> np.ndarray:
    """
    Load audio and return mono waveform at 16 kHz.
    If multi-channel, select the requested channel.
    """
    wav, sr = sf.read(str(audio_path), always_2d=True)  # [T, C]

    if channel < 0 or channel >= wav.shape[1]:
        raise ValueError(
            f"Requested channel={channel}, but audio has {wav.shape[1]} channels: {audio_path}"
        )

    mono = wav[:, channel].astype(np.float32)

    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        up = 16000 // gcd
        down = sr // gcd
        mono = resample_poly(mono, up, down).astype(np.float32)

    return mono


@dataclass
class TranscriptionResult:
    audio_file: str
    transcription: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Whisper ASR over enhanced audio files and save transcriptions to CSV."
    )

    parser.add_argument(
        "--enhanced_dir",
        type=str,
        default=r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\enhanced",
        help="Folder containing enhanced wav files",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="turbo",
        help="Whisper model name",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cpu or cuda",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Whisper language hint",
    )

    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel index if audio is multi-channel",
    )

    parser.add_argument(
        "--max_items",
        type=int,
        default=0,
        help="Optional cap for quick tests; 0 means use all files",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\enhanced",
        help="Directory to store CSV results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    enhanced_dir = Path(args.enhanced_dir)

    if not enhanced_dir.is_dir():
        raise FileNotFoundError(f"Enhanced audio folder not found: {enhanced_dir}")

    enhanced_files = sorted(enhanced_dir.glob("*spk1*.wav"))

    if args.max_items and args.max_items > 0:
        enhanced_files = enhanced_files[:args.max_items]

    if len(enhanced_files) == 0:
        raise FileNotFoundError(f"No enhanced wav files found in: {enhanced_dir}")

    print(f"Found {len(enhanced_files)} enhanced wav files")
    print(f"Loading Whisper model={args.model} on device={args.device}...")

    model = whisper.load_model(args.model, device=args.device)

    results: List[TranscriptionResult] = []

    for idx, audio_path in enumerate(
        tqdm(enhanced_files, desc="Transcribing", unit="utt"),
        start=1,
    ):
        try:
            audio = load_audio_mono(audio_path, args.channel)

            result = model.transcribe(
                audio,
                language=args.language,
                fp16=(args.device == "cuda"),
            )

            hyp_text = result.get("text", "").strip()

            results.append(
                TranscriptionResult(
                    audio_file=audio_path.name,
                    transcription=hyp_text,
                )
            )

            #print(f"[{idx}/{len(enhanced_files)}] {audio_path.name}: {hyp_text}")

        except Exception as e:
            print(f"[{idx}/{len(enhanced_files)}] Failed: {audio_path.name} | {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"enhanced_whisper_{args.model}_transcriptions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "audio_file",
                "transcription",
            ],
        )
        writer.writeheader()

        for row in results:
            writer.writerow(asdict(row))

    print("\n===== TRANSCRIPTION SUMMARY =====")
    print(f"Whisper model: {args.model}, device: {args.device}")
    print(f"Enhanced files requested: {len(enhanced_files)}")
    print(f"Successfully transcribed: {len(results)}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()