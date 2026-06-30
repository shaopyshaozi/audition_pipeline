import socket
import numpy as np
import sounddevice as sd

# ===== match your ODAS raw settings =====
FS = 16000
CHANNELS = 6          # change to 4 if your device is 4-ch
BLOCK = 128
DEVICE_INDEX = 1   # set after you list devices
PORT = 5000
# =======================================

#print(sd.query_devices())
#Set DEVICE_INDEX = 1

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT))
srv.listen(1)

print(f"Listening on 0.0.0.0:{PORT} ... waiting for ODAS to connect")
conn, addr = srv.accept()
print("ODAS connected from:", addr)

def callback(indata, frames, time, status):
    if status:
        print(status)

    # indata: float32, shape (BLOCK, CHANNELS)
    x = np.clip(indata, -1.0, 1.0)
    pcm16 = (x * 32767.0).astype(np.int16)
    level = float(np.mean(np.abs(indata)))
    #print(level)  # 先直接print也行


    try:
        conn.sendall(pcm16.tobytes(order="C"))  # interleaved
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
        print("ODAS disconnected:", e)
        raise sd.CallbackAbort


with sd.InputStream(
    samplerate=FS,
    blocksize=BLOCK,
    channels=CHANNELS,
    dtype="float32",
    device=DEVICE_INDEX,
    callback=callback,
):
    input("Streaming to ODAS. Press ENTER to stop...\n")

conn.close()
srv.close()
