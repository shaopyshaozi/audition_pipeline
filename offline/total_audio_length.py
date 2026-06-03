from pathlib import Path
import soundfile as sf
from tqdm import tqdm

folder = Path("/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/data/dataset_4mic_3spk/Eval/mic")

wav_files = list(folder.glob("*.wav"))

total_seconds = 0.0
file_count = 0

for wav_file in tqdm(wav_files, desc="Reading wav files", unit="file"):
    info = sf.info(str(wav_file))
    total_seconds += info.frames / info.samplerate
    file_count += 1

print(f"Files: {file_count}")
print(f"Total seconds: {total_seconds:.2f}")
print(f"Total minutes: {total_seconds / 60:.2f}")
print(f"Total hours: {total_seconds / 3600:.2f}")