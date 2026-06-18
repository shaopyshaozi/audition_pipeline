import pickle
import numpy as np
import soundfile as sf

npz_path = "/mnt/e/generated_data/train/0.npz"
audio_path = "/mnt/e/generated_data/train/0.wav"

audio, fs = sf.read(audio_path)

print("\nAudio info:")
print("Audio shape:", audio.shape)
print("Audio sampling rate:", fs)
print("Audio duration:", audio.shape[0] / fs, "seconds")

with open(npz_path, "rb") as f:
    data = pickle.load(f)

print("\nKeys saved in file:")
print(data.keys())

# New transformed label
doaw = data["DOAw"]

print("\nDOA info:")
print("DOAw shape:", doaw.shape)  # expected: (20, 2, 3)

azimuth_deg = np.rad2deg(doaw[:, 1, :])
elevation_deg = np.rad2deg(doaw[:, 0, :])

print("Segmented elevation range:", elevation_deg.min(), elevation_deg.max())
print("Segmented azimuth range per source:")

for i in range(azimuth_deg.shape[1]):
    print(f"Source {i}: {azimuth_deg[:, i].min():.2f}° to {azimuth_deg[:, i].max():.2f}°")

print("\nFirst segmented-frame azimuths:", azimuth_deg[0])
print("Last segmented-frame azimuths:", azimuth_deg[-1])

print("\nAll segmented azimuths:")
for i in range(azimuth_deg.shape[1]):
    print(f"Source {i}:")
    print(np.round(azimuth_deg[:, i], 2))

if "mic_vad_sources" in data:
    print("\nVAD info:")
    print("mic_vad_sources shape:", data["mic_vad_sources"].shape)

if "dp_vad" in data:
    print("dp_vad shape:", data["dp_vad"].shape)