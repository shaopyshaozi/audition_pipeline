"""
Offline sequential SSL -> DSE -> ASR realtime benchmark.

Loads IPDNet2, DSENet, and Whisper once, then processes each saved
multichannel wav one scene at a time. For each scene:

1. Run SSL once on the representative multichannel mixture.
2. Convert SSL output to up to three DOAs.
3. Run DSENet over the three DOA inputs in configurable chunks.
4. Select the loudest enhanced output and run Whisper once on that stream.
5. Score the transcript against the ground-truth spk1 text for the scene.

DSENet is chunked with a default batch size of 1 so long scenes do not need to
enhance all candidate DOAs in one forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import string
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
DSENET_DATA_ROOT = PROJECT_ROOT / "data" / "dataset_4mic_3spk"

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


def circular_angle_error_deg(pred_deg: float, gt_deg: float) -> float:
    return float(abs((pred_deg - gt_deg + 180.0) % 360.0 - 180.0))


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def edit_distance_words(ref_words: Sequence[str], hyp_words: Sequence[str]) -> int:
    n = len(ref_words)
    m = len(hyp_words)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if len(ref_words) == 0:
        return (0.0 if len(hyp_words) == 0 else 1.0, len(hyp_words), 0)
    dist = edit_distance_words(ref_words, hyp_words)
    return dist / len(ref_words), dist, len(ref_words)


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


def load_dominant_spk1_doas(clean_dir: Path) -> Dict[int, int]:
    dominant_doas: Dict[int, int] = {}
    for clean_path in sorted(clean_dir.glob("clean_fileid_*_doa*_spk1.wav")):
        fileid = parse_fileid(clean_path)
        dominant_doas[fileid] = parse_doa(clean_path)
    return dominant_doas


def load_dominant_spk1_texts(text_dir: Path) -> Dict[int, Path]:
    dominant_texts: Dict[int, Path] = {}
    for text_path in sorted(text_dir.glob("text_fileid_*_doa*_spk1.txt")):
        fileid = parse_fileid(text_path)
        dominant_texts[fileid] = text_path
    return dominant_texts


def signal_rms(sig: np.ndarray) -> float:
    sig64 = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(sig64)) + 1e-12))


def select_loudest_enhanced(enhanced_batch: Sequence[np.ndarray]) -> Tuple[int, float]:
    if not enhanced_batch:
        raise ValueError("Cannot select loudest enhanced signal from an empty batch.")
    rms_values = [signal_rms(enhanced) for enhanced in enhanced_batch]
    selected_idx = int(np.argmax(rms_values))
    return selected_idx, rms_values[selected_idx]


def postprocess_doa_from_tensors(
    doa_est: torch.Tensor,
    vad_est: torch.Tensor,
    num_sources: int,
    vad_th: float,
) -> List[int]:
    """
    Cluster active SSL DOA points into exactly num_sources DOAs.

    This follows the original clustering approach, but does not remove duplicate
    rounded DOAs and does not discard weak/small clusters. That keeps a fixed
    three-DOA output for the downstream DSENet batch whenever enough active SSL
    points exist to form three clusters.
    """
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

    labels = KMeans(n_clusters=num_sources, random_state=0, n_init=10).fit_predict(
        xy,
        sample_weight=valid_weights_np,
    )

    final_doas: List[int] = []
    for source_id in range(num_sources):
        cluster_angles = valid_angles_np[labels == source_id]
        cluster_weights = valid_weights_np[labels == source_id]
        final_doas.append(int(round(circular_mean_deg_360(cluster_angles, cluster_weights))) % 360)

    return sorted(final_doas)


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
    selected_enhanced_index: int
    selected_doa: int
    selected_rms: float
    selected_enhanced_file: str
    gt_dominant_spk1_doa: Optional[int]
    selected_doa_error_deg: Optional[float]
    gt_text_file: str
    wer: Optional[float]
    edit_distance: Optional[int]
    ref_words: Optional[int]
    reference: str
    hypothesis: str
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
    parser.add_argument("--mic_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "mic")
    parser.add_argument("--clean_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "clean")
    parser.add_argument("--text_dir", type=Path, default=DSENET_DATA_ROOT / "Eval" / "text")
    parser.add_argument("--ipd_ckpt", type=Path, default=SSL_ROOT / "checkpoints" / "ipdnet2_23.ckpt")
    parser.add_argument("--dse_ckpt", type=Path, default=DSE_ROOT / "DSE_96.ckpt")
    parser.add_argument("--out_dir", type=Path, default=OFFLINE_ROOT / "results")
    parser.add_argument("--whisper_model", type=str, default="small")
    parser.add_argument("--whisper_device", type=str, default="cuda")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--vad_th", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=30)
    parser.add_argument(
        "--dse_batch_size",
        type=int,
        default=1,
        help="How many candidate DOAs to enhance per DSENet forward pass. Use 1 for long audio.",
    )
    parser.add_argument("--max_items", type=int, default=0, help="Limit mic wav entries for a quick test; 0 means all.")
    parser.add_argument("--save_enhanced", action="store_true", help="Save the selected loudest enhanced wav.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.mic_dir.is_dir():
        raise FileNotFoundError(f"Mic folder not found: {args.mic_dir}")
    if not args.clean_dir.is_dir():
        raise FileNotFoundError(f"Clean folder not found: {args.clean_dir}")
    if not args.text_dir.is_dir():
        raise FileNotFoundError(f"Text folder not found: {args.text_dir}")
    if args.whisper_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Whisper was requested on CUDA, but torch.cuda.is_available() is False.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir = args.out_dir / "pipeline_realtime_enhanced"
    if args.save_enhanced:
        enhanced_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Whisper device: {args.whisper_device}")
    print(f"DSENet batch size: {args.dse_batch_size}")
    print(f"Mic input folder: {args.mic_dir}")
    print(f"Dominant speaker clean folder: {args.clean_dir}")
    print(f"Dominant speaker text folder: {args.text_dir}")
    print("Loading IPDNet2 once...")
    ipd_model, doa_decoder = load_ipdnet2(args.ipd_ckpt, args.device)
    print("Loading DSENet once...")
    dse_model = load_dsenet(args.dse_ckpt, args.device)
    print(f"Loading Whisper once: {args.whisper_model} on {args.whisper_device}")
    whisper_model = whisper.load_model(args.whisper_model, device=args.whisper_device)

    target_files = unique_mic_files(args.mic_dir, args.max_items)
    grouped = group_targets_by_fileid(target_files)
    dominant_spk1_doas = load_dominant_spk1_doas(args.clean_dir)
    dominant_spk1_texts = load_dominant_spk1_texts(args.text_dir)
    print(f"Selected mic wav entries: {len(target_files)}")
    print(f"Unique fileid groups: {len(grouped)}")
    print(f"Dominant spk1 DOA references: {len(dominant_spk1_doas)}")
    print(f"Dominant spk1 text references: {len(dominant_spk1_texts)}")

    scene_results: List[SceneTiming] = []
    skipped_no_doa = 0
    missing_dominant_gt = 0
    missing_text = 0
    total_edits = 0
    total_ref_words = 0

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
        )

        if len(pred_doas) != args.num_sources:
            skipped_no_doa += 1
            print(
                f"fileid={fileid}: expected {args.num_sources} SSL DOAs, "
                f"got {len(pred_doas)}, skipped."
            )
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

        selected_idx, selected_rms = select_loudest_enhanced(enhanced_batch)
        selected_doa = pred_doas[selected_idx]
        selected_save_name = f"enhanced_fileid_{fileid}_pred{selected_doa}_idx{selected_idx}_loudest.wav"
        enhanced_for_asr = enhanced_batch[selected_idx]
        gt_dominant_doa = dominant_spk1_doas.get(fileid)
        selected_doa_error = (
            circular_angle_error_deg(selected_doa, gt_dominant_doa)
            if gt_dominant_doa is not None
            else None
        )
        if gt_dominant_doa is None:
            missing_dominant_gt += 1

        text_path = dominant_spk1_texts.get(fileid)
        ref_text = ""
        hyp_text = ""
        sample_wer: Optional[float] = None
        dist: Optional[int] = None
        ref_word_count: Optional[int] = None
        if text_path is None:
            missing_text += 1

        if args.save_enhanced:
            sf.write(str(enhanced_dir / selected_save_name), enhanced_for_asr, sr)

        def run_asr():
            return whisper_model.transcribe(
                enhanced_for_asr,
                language=args.language,
                fp16=args.whisper_device.startswith("cuda"),
            )

        asr_out, whisper_sec = elapsed_seconds(args.whisper_device, run_asr)
        hyp_text = asr_out.get("text", "").strip()
        if text_path is not None:
            ref_text = text_path.read_text(encoding="utf-8").strip()
            sample_wer, dist, ref_word_count = wer(ref_text, hyp_text)
            total_edits += dist
            total_ref_words += ref_word_count

        total_sec = ipd_sec + dse_sec + whisper_sec
        scene_results.append(
            SceneTiming(
                fileid=fileid,
                mic_file=mic_path.name,
                duration_sec=duration_sec,
                predicted_doa_count=len(pred_doas),
                predicted_doas=",".join(str(doa) for doa in pred_doas),
                selected_enhanced_index=selected_idx,
                selected_doa=selected_doa,
                selected_rms=selected_rms,
                selected_enhanced_file=selected_save_name,
                gt_dominant_spk1_doa=gt_dominant_doa,
                selected_doa_error_deg=selected_doa_error,
                gt_text_file=text_path.name if text_path is not None else "",
                wer=sample_wer,
                edit_distance=dist,
                ref_words=ref_word_count,
                reference=ref_text,
                hypothesis=hyp_text,
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

    details_csv = args.out_dir / f"pipeline_whisper_{args.whisper_model}_wer_details_1asr.csv"
    with details_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(SceneTiming.__dataclass_fields__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in scene_results:
            writer.writerow(asdict(row))

    selected_doa_errors = [
        r.selected_doa_error_deg
        for r in scene_results
        if r.selected_doa_error_deg is not None
    ]
    sample_wers = [r.wer for r in scene_results if r.wer is not None]

    summary = {
        "mic_dir": str(args.mic_dir),
        "clean_dir": str(args.clean_dir),
        "text_dir": str(args.text_dir),
        "ipd_ckpt": str(args.ipd_ckpt),
        "dse_ckpt": str(args.dse_ckpt),
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "device": args.device,
        "dse_batch_size": args.dse_batch_size,
        "selection": "loudest_enhanced_rms",
        "selected_mic_wav_entries": len(target_files),
        "unique_fileid_groups": len(grouped),
        "dominant_spk1_doa_references": len(dominant_spk1_doas),
        "dominant_spk1_text_references": len(dominant_spk1_texts),
        "evaluated_scenes": len(scene_results),
        "skipped_no_doa": skipped_no_doa,
        "missing_dominant_gt": missing_dominant_gt,
        "missing_text": missing_text,
        "corpus_wer": (total_edits / total_ref_words) if total_ref_words > 0 else 0.0,
        "mean_sample_wer": float(np.mean(sample_wers)) if sample_wers else 0.0,
        "mean_selected_doa_error_deg": float(np.mean(selected_doa_errors)) if selected_doa_errors else 0.0,
        "median_selected_doa_error_deg": float(np.median(selected_doa_errors)) if selected_doa_errors else 0.0,
        "p95_selected_doa_error_deg": float(np.percentile(selected_doa_errors, 95)) if selected_doa_errors else 0.0,
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

    summary_json = args.out_dir / f"pipeline_whisper_{args.whisper_model}_wer_summary_1asr.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== REALTIME SUMMARY =====")
    print(f"Evaluated scenes: {summary['evaluated_scenes']}")
    print(f"Under realtime (< audio duration) count: {summary['under_realtime_count']}")
    print(f"Under realtime rate: {summary['under_realtime_rate']:.4f}")
    print(f"Missing dominant spk1 DOA references: {summary['missing_dominant_gt']}")
    print(f"Missing dominant spk1 text references: {summary['missing_text']}")
    print(f"Corpus WER vs spk1 text: {summary['corpus_wer']:.4f}")
    print(f"Mean sample WER vs spk1 text: {summary['mean_sample_wer']:.4f}")
    print(
        "Selected loudest DOA error vs spk1 GT: "
        f"mean={summary['mean_selected_doa_error_deg']:.2f} deg, "
        f"median={summary['median_selected_doa_error_deg']:.2f} deg, "
        f"p95={summary['p95_selected_doa_error_deg']:.2f} deg"
    )
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
