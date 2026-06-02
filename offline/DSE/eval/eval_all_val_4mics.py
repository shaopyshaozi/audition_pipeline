import os
import re
import sys
import torch
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from DOATrainer import TrainModule
from models.arch.DSENet import DSENet
from models.utils.metrics import cal_metrics_functional, recover_scale
from tqdm import tqdm


def load_audio_file(path: str, target_sr: int):
    wav, sr = sf.read(path, always_2d=True)   # [T, C]
    wav = wav.T.astype(np.float32)            # [C, T]

    if sr != target_sr:
        resampled = []
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        for ch in wav:
            resampled_ch = resample_poly(ch, up, down).astype(np.float32)
            resampled.append(resampled_ch)
        wav = np.stack(resampled, axis=0)
        sr = target_sr

    return torch.from_numpy(wav), sr


def parse_doa_width(filename: str):
    doa_match = re.search(r"doa(\d+)", filename)
    width_match = re.search(r"width(\d+)", filename)

    doa_val = int(doa_match.group(1)) if doa_match else 0
    width_val = int(width_match.group(1)) if width_match else 30
    return doa_val, width_val


def build_noisy_name_from_clean(clean_name: str, dataset_tag: str):
    # clean_fileid_0_doa104_spk6.wav
    # clean_fileid_0_doa104_width40_spk6.wav
    parts = clean_name.split("_")

    if len(parts) < 5:
        raise ValueError(f"Unexpected clean filename format: {clean_name}")

    if parts[4].startswith("width"):
        noisy_name = f"mic_{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}_{dataset_tag}.wav"
    else:
        noisy_name = f"mic_{parts[1]}_{parts[2]}_{parts[3]}_{dataset_tag}.wav"

    return noisy_name


def scalar_only(d):
    out = {}
    for k, v in d.items():
        if k.endswith("_all"):
            continue
        if v is None:
            continue
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                out[k] = float(v.item())
        elif isinstance(v, np.ndarray):
            if v.size == 1:
                out[k] = float(v.item())
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
    return out


def main():
    ckpt_path = r"/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/DSE/DSE_96.ckpt"

    clean_dir = r"/mnt/d/邵鹏远/UCL/博1/code/DSENet/data/dataset_4mic_6spk/test/clean"
    noisy_dir = r"/mnt/d/邵鹏远/UCL/博1/code/DSENet/data/dataset_4mic_6spk/test/mic"
    save_enhanced_dir = r"/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/offline/DSE/eval/enhanced_val_4mics"

    os.makedirs(save_enhanced_dir, exist_ok=True)

    sample_rate = 16000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    arch = DSENet(
        dim_input=8,
        dim_output=2,
        dim_squeeze=8,
        num_layers=8,
        num_freqs=129,
        encoder_kernel_size=5,
        dim_hidden=192,
        dim_ffn=192,
        num_heads=4,
        dropout=(0.0, 0.0, 0.0),
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        padding="zeros",
        full_share=0,
        d_embedding=40,
        d_alpha=20,
        width_emb_dim=3,
        width_stage=15,
        width_control=True,
    )

    print("Loading checkpoint...")
    model = TrainModule.load_from_checkpoint(
        ckpt_path,
        arch=arch,
        map_location=device
    )
    model.eval()
    model.to(device)
    model.float()

    dataset_tag = "6spk"
    print("Dataset tag:", dataset_tag)

    clean_files = sorted(
        [f for f in os.listdir(clean_dir) if f.lower().endswith(".wav")]
    )

    print(f"Found {len(clean_files)} clean files")

    metrics_list = ["SDR", "SI_SDR", "WB_PESQ"]

    all_rows = []
    aggregate = {}

    for idx, clean_name in enumerate(tqdm(clean_files), 1):
        try:
            clean_path = os.path.join(clean_dir, clean_name)
            noisy_name = build_noisy_name_from_clean(clean_name, dataset_tag)
            noisy_path = os.path.join(noisy_dir, noisy_name)

            if not os.path.exists(noisy_path):
                print(f"[{idx}/{len(clean_files)}] Missing noisy file: {noisy_name}")
                continue

            clean_ds, _ = load_audio_file(clean_path, sample_rate)
            noisy_ds, _ = load_audio_file(noisy_path, sample_rate)

            if clean_ds.shape[1] != noisy_ds.shape[1]:
                print(f"[{idx}/{len(clean_files)}] Length mismatch: {clean_name}")
                continue

            if clean_ds.shape[0] == 1 and noisy_ds.shape[0] > 1:
                clean_ds = clean_ds.repeat(noisy_ds.shape[0], 1)

            x = noisy_ds.unsqueeze(0).float().to(device)   # [1, C, T]
            yr = clean_ds.unsqueeze(0).float().to(device)  # [1, C, T]

            doa_value, width_value = parse_doa_width(clean_name)
            DOA = torch.tensor([doa_value], dtype=torch.long, device=device)
            width = torch.tensor([width_value], dtype=torch.long, device=device)

            ref_channel = model.ref_channel

            with torch.no_grad():
                yr_hat = model.forward(x, DOA, width)           # [1, 1, T]
                yr_ref = yr[:, ref_channel, :].unsqueeze(1)     # [1, 1, T]
                x_ref = x[:, ref_channel, :].unsqueeze(1)       # [1, 1, T]

                if model.loss.is_scale_invariant_loss:
                    yr_hat = recover_scale(
                        preds=yr_hat,
                        mixture=x[:, ref_channel, :],
                        scale_src_together=True,
                        norm_if_exceed_1=False
                    )

            metrics, input_metrics, imp_metrics = cal_metrics_functional(
                metrics_list,
                yr_hat[0],
                yr_ref[0],
                x_ref[0],
                sample_rate,
                device_only= None
            )

            metrics_s = scalar_only(metrics)
            input_metrics_s = scalar_only(input_metrics)
            imp_metrics_s = scalar_only(imp_metrics)

            row = {
                "clean_name": clean_name,
                "noisy_name": noisy_name,
                "doa": doa_value,
                "width": width_value,
            }
            row.update(input_metrics_s)
            row.update(metrics_s)
            row.update(imp_metrics_s)
            all_rows.append(row)

            for k, v in row.items():
                if isinstance(v, (int, float, np.floating)) and k not in ["doa", "width"]:
                    aggregate.setdefault(k, []).append(float(v))

            enhanced = yr_hat[0, 0].detach().cpu().numpy()
            save_name = clean_name.replace("clean_", "enhanced_")
            save_path = os.path.join(save_enhanced_dir, save_name)
            sf.write(save_path, enhanced, sample_rate)

            print(f"[{idx}/{len(clean_files)}] Done: {clean_name}")

        except Exception as e:
            print(f"[{idx}/{len(clean_files)}] Failed: {clean_name} | {e}")

    print("\n===== Average Metrics Over Validation Set =====")
    for k in sorted(aggregate.keys()):
        vals = aggregate[k]
        if len(vals) > 0:
            print(f"{k}: {np.mean(vals):.4f}")

    csv_path = os.path.join(save_enhanced_dir, "val_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        if len(all_rows) > 0:
            headers = list(all_rows[0].keys())
            f.write(",".join(headers) + "\n")
            for row in all_rows:
                line = []
                for h in headers:
                    line.append(str(row.get(h, "")))
                f.write(",".join(line) + "\n")

    print(f"\nSaved enhanced wavs to: {save_enhanced_dir}")
    print(f"Saved metrics csv to: {csv_path}")


if __name__ == "__main__":
    main()