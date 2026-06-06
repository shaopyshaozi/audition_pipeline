"""
Offline sequential SSL -> DSE -> ASR realtime benchmark.

Loads IPDNet2, DSENet, and Whisper once, then processes each saved
multichannel 4 s wav one scene at a time. For each scene:

1. Run SSL once on the representative multichannel mixture.
2. Convert SSL output to up to three DOAs.
3. Run DSENet once with a batched DOA input.
4. Run Whisper once on one enhanced output to measure end-to-end latency.

This script is timing-only. It does not require ground-truth text.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import resample_poly
from sklearn.cluster import KMeans
from tqdm import tqdm


OFFLINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = OFFLINE_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "Models"
SSL_ROOT = MODELS_ROOT / "SSL"
DSE_ROOT = MODELS_ROOT / "DSE"
DSENET_DATA_ROOT = PROJECT_ROOT/ "data" / "dataset_4mic_3spk_4s"

sys.path.insert(0, str(SSL_ROOT))
from IPDnet2_3spk import OnlineSpatialNet  # noqa: E402
import Module_3spk as ssl_module  # noqa: E402
from utils_ import audiowu_high_array_geometry, forgetting_norm  # noqa: E402

sys.path.insert(0, str(DSE_ROOT))
from DOATrainer import TrainModule  # noqa: E402
from models.arch.DSENet import DSENet  # noqa: E402
from models.utils.metrics import recover_scale  # noqa: E402


def cuda_sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_seconds(device: str, fn):
    cuda_sync(device)
    start = time.perf_counter()
    value = fn()
    cuda_sync(device)
    return value, time.perf_counter() - start


def circular_mean_deg_360(angles_deg: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    angles_deg = np.asarray(angles_deg) % 360.0
    angles_rad = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(angles_rad)
    x = np.sum(weights * np.cos(angles_rad))
    y = np.sum(weights * np.sin(angles_rad))
    return float(np.rad2deg(np.arctan2(y, x)) % 360.0)


def parse_fileid(path_or_name: Path | str) -> int:
    match = re.search(r"fileid_(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse fileid from: {path_or_name}")
    return int(match.group(1))


def parse_doa(path_or_name: Path | str) -> int:
    match = re.search(r"doa(\d+)", Path(path_or_name).name)
    if not match:
        raise ValueError(f"Could not parse doa from: {path_or_name}")
    return int(match.group(1))


def load_multichannel_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), always_2d=True)
    wav = wav.astype(np.float32)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = np.stack(
            [resample_poly(wav[:, ch], up, down).astype(np.float32) for ch in range(wav.shape[1])],
            axis=1,
        )
        sr = target_sr
    return wav, sr


def unique_mic_files(mic_dir: Path, max_items: int) -> List[Path]:
    all_files = sorted(mic_dir.glob("*.wav"), key=lambda p: (parse_fileid(p), parse_doa(p), p.name))
    if max_items > 0:
        return all_files[:max_items]
    return all_files


def group_targets_by_fileid(mic_files: Iterable[Path]) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in mic_files:
        grouped.setdefault(parse_fileid(path), []).append(path)
    return grouped


def choose_representative_mic(target_paths: Sequence[Path]) -> Path:
    return sorted(target_paths, key=lambda p: (parse_doa(p), p.name))[0]


def postprocess_doa_from_tensors(
    doa_est: torch.Tensor,
    vad_est: torch.Tensor,
    num_sources: int,
    vad_th: float,
    min_points_per_source: int,
) -> List[int]:
    doa_np = doa_est.detach().cpu().numpy()
    vad_np = vad_est.detach().cpu().numpy()

    azi = doa_np[0, :, 1, :] % 360.0
    score = vad_np[0, :, :]
    active = score < vad_th

    valid_angles = []
    valid_weights = []
    for t in range(azi.shape[0]):
        for k in range(azi.shape[1]):
            if active[t, k]:
                valid_angles.append(azi[t, k])
                valid_weights.append(1.0 / (score[t, k] + 1e-6))

    if len(valid_angles) < num_sources:
        return []

    valid_angles_np = np.asarray(valid_angles, dtype=np.float32)
    valid_weights_np = np.asarray(valid_weights, dtype=np.float32)
    angle_rad = np.deg2rad(valid_angles_np)
    xy = np.stack([np.cos(angle_rad), np.sin(angle_rad)], axis=1)

    n_clusters = min(num_sources, len(valid_angles_np))
    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(
        xy,
        sample_weight=valid_weights_np,
    )

    final_doas: List[int] = []
    for source_id in range(n_clusters):
        cluster_angles = valid_angles_np[labels == source_id]
        cluster_weights = valid_weights_np[labels == source_id]
        if len(cluster_angles) < min_points_per_source:
            continue
        final_doas.append(int(round(circular_mean_deg_360(cluster_angles, cluster_weights))) % 360)

    return sorted(set(final_doas))[:num_sources]


def torch_load_checkpoint(path: Path, device: str):
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


class IPDNet2Inference(torch.nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device_name = device
        self.arch = OnlineSpatialNet(
            dim_input=8,
            dim_output=18,
            num_layers=8,
            dim_hidden=96,
            num_heads=4,
            kernel_size=(5, 3),
            conv_groups=(8, 8),
            norms=["LN", "LN", "GN", "LN", "LN", "LN"],
            dim_squeeze=8,
            num_freqs=256,
            attention="mamba(16,4)",
            rope=False,
            time_compression_layer=0,
            fre_compression_ratio=16,
            time_compression_ratio=5,
        )
        self.dostft = ssl_module.STFT(win_len=512, win_shift_ratio=0.625, nfft=512)
        self.fre_range_used = range(1, 257)

    def forward(self, mic_sig_batch: torch.Tensor) -> torch.Tensor:
        in_batch = self.data_preprocess_inference(mic_sig_batch)[0]
        return self.arch(in_batch)

    def data_preprocess_inference(self, mic_sig_batch: torch.Tensor, eps: float = 1e-6) -> List[torch.Tensor]:
        stft = self.dostft(signal=mic_sig_batch)
        stft_rebatch = stft.permute(0, 3, 1, 2).to(self.device_name)
        mag = torch.abs(stft_rebatch)
        mean_value = forgetting_norm(mag, sample_length=249)
        stft_rebatch_real = torch.real(stft_rebatch) / (mean_value + eps)
        stft_rebatch_imag = torch.imag(stft_rebatch) / (mean_value + eps)
        real_imag_batch = torch.cat((stft_rebatch_real, stft_rebatch_imag), dim=1)
        return [real_imag_batch[:, :, self.fre_range_used, :]]


def load_ipdnet2(ckpt_path: Path, device: str) -> Tuple[IPDNet2Inference, torch.nn.Module]:
    model = IPDNet2Inference(device=device)
    ckpt = torch_load_checkpoint(ckpt_path, device)
    state_dict = ckpt.get("state_dict", ckpt)
    arch_state = {
        key.replace("arch.", "", 1): value
        for key, value in state_dict.items()
        if key.startswith("arch.")
    }
    missing, unexpected = model.arch.load_state_dict(arch_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected IPDNet2 checkpoint keys: {unexpected[:5]}")
    if missing:
        print(f"Warning: IPDNet2 missing {len(missing)} arch keys while loading checkpoint.")

    model.eval().to(device)

    use_mic_id = [2, 4, 6, 8]
    mic_location = audiowu_high_array_geometry()[use_mic_id]
    doa_decoder = ssl_module.PredDOA_Inference(
        mic_location=mic_location,
        max_track=3,
        max_num_sources=1,
        dev=device,
    )
    doa_decoder.eval().to(device)
    return model, doa_decoder


def load_dsenet(ckpt_path: Path, device: str) -> TrainModule:
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
    model = TrainModule.load_from_checkpoint(str(ckpt_path), arch=arch, map_location=device)
    model.eval().to(device).float()
    return model


def enhance_doa_batch(
    dse_model: TrainModule,
    noisy_ct: torch.Tensor,
    doa_values: Sequence[int],
    width_value: int,
    device: str,
) -> List[np.ndarray]:
    batch_size = len(doa_values)
    if batch_size == 0:
        return []

    x = noisy_ct.unsqueeze(0).repeat(batch_size, 1, 1).float().to(device)
    doa = torch.tensor(doa_values, dtype=torch.long, device=device)
    width = torch.full((batch_size,), width_value, dtype=torch.long, device=device)

    with torch.inference_mode():
        yr_hat = dse_model.forward(x, doa, width)
        if dse_model.loss.is_scale_invariant_loss:
            yr_hat = recover_scale(
                preds=yr_hat,
                mixture=x[:, dse_model.ref_channel, :],
                scale_src_together=True,
                norm_if_exceed_1=False,
            )

    return [yr_hat[idx, 0].detach().cpu().numpy().astype(np.float32) for idx in range(batch_size)]


@dataclass
class SceneTiming:
    fileid: int
    mic_file: str
    duration_sec: float
    predicted_doa_count: int
    predicted_doas: str
    ipdnet2_sec: float
    dsenet_sec: float
    whisper_sec: float
    total_sec: float
    ipdnet2_rtf: float
    dsenet_rtf: float
    whisper_rtf: float
    total_rtf: float
    under_realtime: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IPDNet2 -> DSENet -> Whisper realtime benchmark.")
    parser.add_argument("--mic_dir", type=Path, default=DSENET_DATA_ROOT / "test" / "mic")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "checkpoints" / "ipdnet2_23.ckpt")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_96.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results")
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--vad_th", type=float, default=0.2)
    parser.add_argument("--min_points_per_source", type=int, default=3)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument("--dse_batch_size", type=int, default=3)
    parser.add_argument("--whisper_index", type=int, default=0, help="Which enhanced stream to pass to Whisper.")
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_realtime_enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"DSENet batch size: {args.dse_batch_size}")
    print(f"Mic input folder: {args.mic_dir}")
    print("Loading IPDNet2 once...")
    ipd_model, doa_decoder = load_ipdnet2(args.ipd_ckpt, args.device)
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")

    scene_results: List[SceneTiming] = []
    skipped_no_doa = 0

    for fileid, target_paths in tqdm(grouped.items(), desc="Realtime", unit="scene"):
        mic_path = choose_representative_mic(target_paths)
        wav_tc, sr = load_multichannel_audio(mic_path, target_sr=args.sample_rate)
        duration_sec = wav_tc.shape[0] / float(sr)
        mic_batch = torch.from_numpy(wav_tc).unsqueeze(0)
        noisy_ct = torch.from_numpy(wav_tc.T.copy())

        def run_ssl():
            with torch.inference_mode():
                pred_ipd = ipd_model(mic_batch)
                return doa_decoder(pred_ipd)

        ssl_out, ipd_sec = elapsed_seconds(args.device, run_ssl)
        pred_doas = postprocess_doa_from_tensors(
            ssl_out["doa_est"],
            ssl_out["vad_est"],
            num_sources=args.num_sources,
            vad_th=args.vad_th,
            min_points_per_source=args.min_points_per_source,
        )

        if not pred_doas:
            skipped_no_doa += 1
            print(f"fileid={fileid}: no usable SSL DOA, skipped.")
            continue

        dse_batch_size = len(pred_doas) if args.dse_batch_size <= 0 else args.dse_batch_size
        enhanced_batch: List[np.ndarray] = []
        dse_sec = 0.0

        for start in range(0, len(pred_doas), dse_batch_size):
            doa_chunk = pred_doas[start:start + dse_batch_size]
            enhanced_chunk, dse_chunk_sec = elapsed_seconds(
                args.device,
                lambda doa_chunk=doa_chunk: enhance_doa_batch(
                    dse_model,
                    noisy_ct,
                    doa_chunk,
                    args.width,
                    args.device,
                ),
            )
            enhanced_batch.extend(enhanced_chunk)
            dse_sec += dse_chunk_sec
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not enhanced_batch:
            skipped_no_doa += 1
            print(f"fileid={fileid}: DSENet produced no enhanced output, skipped.")
            continue

        if args.save_enhanced:
            for idx, (pred_doa, enhanced) in enumerate(zip(pred_doas, enhanced_batch)):
                save_name = f"enhanced_fileid_{fileid}_pred{pred_doa}_idx{idx}.wav"
                sf.write(str(enhanced_dir / save_name), enhanced, sr)

        whisper_idx = min(max(args.whisper_index, 0), len(enhanced_batch) - 1)
        enhanced_for_asr = enhanced_batch[whisper_idx]

        def run_asr():
            return whisper_model.transcribe(
                enhanced_for_asr,
                language=args.language,
                fp16=args.whisper_device.startswith("cuda"),
            )

        _, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
        total_sec = ipd_sec + dse_sec + whisper_sec
        scene_results.append(
            SceneTiming(
                fileid=fileid,
                mic_file=mic_path.name,
                duration_sec=duration_sec,
                predicted_doa_count=len(pred_doas),
                predicted_doas=",".join(str(doa) for doa in pred_doas),
                ipdnet2_sec=ipd_sec,
                dsenet_sec=dse_sec,
                whisper_sec=whisper_sec,
                total_sec=total_sec,
                ipdnet2_rtf=ipd_sec / duration_sec,
                dsenet_rtf=dse_sec / duration_sec,
                whisper_rtf=whisper_sec / duration_sec,
                total_rtf=total_sec / duration_sec,
                under_realtime=int(total_sec < duration_sec),
            )
        )

    details_csv = args.out_dir / f"pipeline_realtime_{args.whisper_model}_details.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    summary = {
        "mic_dir": str(args.mic_dir),
        "ipd_ckpt": str(args.ipd_ckpt),
        "dse_ckpt": str(args.dse_ckpt),
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "dse_batch_size": args.dse_batch_size,
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "evaluated_scenes": len(scene_results),
        "skipped_no_doa": skipped_no_doa,
        "mean_duration_sec": float(np.mean([r.duration_sec for r in scene_results])) if scene_results else 0.0,
        "mean_ipdnet2_sec": float(np.mean([r.ipdnet2_sec for r in scene_results])) if scene_results else 0.0,
        "mean_dsenet_sec": float(np.mean([r.dsenet_sec for r in scene_results])) if scene_results else 0.0,
        "mean_whisper_sec": float(np.mean([r.whisper_sec for r in scene_results])) if scene_results else 0.0,
        "mean_total_sec": float(np.mean([r.total_sec for r in scene_results])) if scene_results else 0.0,
        "mean_total_rtf": float(np.mean([r.total_rtf for r in scene_results])) if scene_results else 0.0,
        "median_total_sec": float(np.median([r.total_sec for r in scene_results])) if scene_results else 0.0,
        "p95_total_sec": float(np.percentile([r.total_sec for r in scene_results], 95)) if scene_results else 0.0,
        "under_realtime_count": int(sum(r.under_realtime for r in scene_results)),
        "under_realtime_rate": float(np.mean([r.under_realtime for r in scene_results])) if scene_results else 0.0,
    }

    summary_json = args.out_dir / f"pipeline_realtime_{args.whisper_model}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== REALTIME SUMMARY =====")
    print(f"Evaluated scenes: {summary['evaluated_scenes']}")
    print(f"Under realtime (< audio duration) count: {summary['under_realtime_count']}")
    print(f"Under realtime rate: {summary['under_realtime_rate']:.4f}")
    print(
        "Mean timing per scene: "
        f"IPDNet2={summary['mean_ipdnet2_sec']:.3f}s, "
        f"DSENet={summary['mean_dsenet_sec']:.3f}s, "
        f"Whisper={summary['mean_whisper_sec']:.3f}s, "
        f"total={summary['mean_total_sec']:.3f}s"
    )
    print(f"Mean total RTF: {summary['mean_total_rtf']:.3f}")
    print(f"P95 total time: {summary['p95_total_sec']:.3f}s")
    print(f"Saved realtime details: {details_csv}")
    print(f"Saved summary: {summary_json}")


if __name__ == "__main__":
    main()
