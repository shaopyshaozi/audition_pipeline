import pyaudio
import wave
import numpy as np

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6      # ReSpeaker 6-channel firmware input
RESPEAKER_WIDTH = 2         # 2 bytes = int16
RESPEAKER_INDEX = 1         # change based on getDeviceInfo.py
CHUNK = 1024
RECORD_SECONDS = 5

WAVE_OUTPUT_FILENAME = "output_raw_4ch.wav"

p = pyaudio.PyAudio()

stream = p.open(
    rate=RESPEAKER_RATE,
    format=p.get_format_from_width(RESPEAKER_WIDTH),
    channels=RESPEAKER_CHANNELS,
    input=True,
    input_device_index=RESPEAKER_INDEX,
    frames_per_buffer=CHUNK,
)

print("* recording")

frames = []

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)

    # Convert interleaved 6-channel byte data to int16 array
    audio = np.frombuffer(data, dtype=np.int16)

    # Reshape from:
    # [ch0_s0, ch1_s0, ch2_s0, ch3_s0, ch4_s0, ch5_s0, ch0_s1, ...]
    # to shape: [num_samples, 6]
    audio = audio.reshape(-1, RESPEAKER_CHANNELS)

    # Extract raw microphone channels 1, 2, 3, 4
    # Output shape: [num_samples, 4]
    raw_4ch = audio[:, 1:5]

    # Save as interleaved 4-channel audio
    frames.append(raw_4ch.astype(np.int16).tobytes())

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
wf.setnchannels(4)  # save only 4 raw mic channels
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b"".join(frames))
wf.close()

print(f"Recording saved as {WAVE_OUTPUT_FILENAME}")