from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent.parent
DEFAULT_FOLDER = PROJECT_ROOT / "data" / "dataset_4mic_3spk" / "Eval" / "mic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the total duration of wav files in a folder.")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wav_files = list(args.folder.glob("*.wav"))

    total_seconds = 0.0
    file_count = 0

    for wav_file in tqdm(wav_files, desc="Reading wav files", unit="file"):
        info = sf.info(str(wav_file))
        total_seconds += info.frames / info.samplerate
        file_count += 1

    print(f"Folder: {args.folder}")
    print(f"Files: {file_count}")
    print(f"Total seconds: {total_seconds:.2f}")
    print(f"Total minutes: {total_seconds / 60:.2f}")
    print(f"Total hours: {total_seconds / 3600:.2f}")


if __name__ == "__main__":
    main()
