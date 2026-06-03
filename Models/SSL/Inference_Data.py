import re
import torch
import soundfile as sf

from torch.utils.data import Dataset
from utils_ import search_files
import numpy as np


class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        all_paths = search_files(data_dir, flag='.wav')

        items = []

        for path in all_paths:
            name = path.split('/')[-1]

            match = re.search(r'fileid_(\d+)', name)
            if match is None:
                continue

            fileid = int(match.group(1))
            items.append((fileid, path))

        items = sorted(items, key=lambda x: x[0])

        selected = {}
        for fileid, path in items:
            if fileid not in selected:
                selected[fileid] = path

        self.data_paths = list(selected.values())

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        sig_path = self.data_paths[idx]
        input_mic_signal, fs = sf.read(sig_path)

        input_mic_signal = np.asarray(input_mic_signal, dtype=np.float32)

        return input_mic_signal

        