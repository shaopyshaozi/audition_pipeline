import sounddevice as sd
for i,d in enumerate(sd.query_devices()):
    if d["max_input_channels"]>0:
        print(i, d["name"], d["max_input_channels"])
