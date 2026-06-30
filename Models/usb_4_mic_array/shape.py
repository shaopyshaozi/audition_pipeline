import soundfile as sf
from pathlib import Path

wav_path = Path(r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker\output_raw_4ch.wav")

audio, sr = sf.read(str(wav_path))

print("Sample rate:", sr)
print("Shape:", audio.shape)

if audio.ndim == 1:
    print("Mono audio")
else:
    print(f"Multichannel audio with {audio.shape[1]} channels")