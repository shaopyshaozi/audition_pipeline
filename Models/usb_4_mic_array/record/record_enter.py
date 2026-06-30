import pyaudio
import wave
import numpy as np
import threading
from pathlib import Path

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6
OUTPUT_CHANNELS = 4
RESPEAKER_WIDTH = 2
RESPEAKER_INDEX = 1
CHUNK = 1024

OUTPUT_DIR = Path(r"D:\邵鹏远\UCL\博1\code\usb_4_mic_array\data\Respeaker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WAVE_OUTPUT_FILENAME = OUTPUT_DIR / "output_raw_4ch.wav"

frames = []
stop_event = threading.Event()


def record_audio(stream):
    print("* recording... Press Enter again to stop.")

    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)

        audio = np.frombuffer(data, dtype=np.int16)
        audio = audio.reshape(-1, RESPEAKER_CHANNELS)

        # Extract raw MIC1-MIC4
        raw_4ch = audio[:, 1:5]

        frames.append(raw_4ch.astype(np.int16).tobytes())


p = pyaudio.PyAudio()

stream = p.open(
    rate=RESPEAKER_RATE,
    format=p.get_format_from_width(RESPEAKER_WIDTH),
    channels=RESPEAKER_CHANNELS,
    input=True,
    input_device_index=RESPEAKER_INDEX,
    frames_per_buffer=CHUNK,
)

input("Press Enter to start recording...")

record_thread = threading.Thread(target=record_audio, args=(stream,))
record_thread.start()

input("Press Enter to stop recording...")

stop_event.set()
record_thread.join()

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

with wave.open(str(WAVE_OUTPUT_FILENAME), "wb") as wf:
    wf.setnchannels(OUTPUT_CHANNELS)
    wf.setsampwidth(RESPEAKER_WIDTH)
    wf.setframerate(RESPEAKER_RATE)
    wf.writeframes(b"".join(frames))

print(f"Recording saved as {WAVE_OUTPUT_FILENAME}")