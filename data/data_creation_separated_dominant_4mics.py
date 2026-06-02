#!/usr/bin/env python3
"""
ASR dataset creation with 3 overlapping speakers:
one dominant target speaker and two quieter background speakers.
With ground turth transcripts.
4 MICS Respeaker version---------------------------------------------------------------------
Valid for ASR evaluation only


What this script generates
--------------------------
dataset_root/
    clean/
    mic/
    text/

Filename convention
---------------------------
clean_fileid_<sceneid>_doa<angle>_spk<k>.wav
mic_fileid_<sceneid>_doa<angle>_3spk.wav
text_fileid_<sceneid>_doa<angle>_spk<k>.txt

Notes
-----
- One acoustic scene contains 6 simultaneous speakers + 1 independent noise source.
- From one scene, we generate up to 6 stage-1 training items, one target speaker at a time.
- The same multichannel mixture may be saved multiple times under different DOA-specific
  filenames so it matches the trainer's path reconstruction logic.
- The target ("clean") file is saved as MONO: the reverberant target speaker at the reference mic.
  This is acceptable because the trainer repeats mono clean to match channels if needed.
- Width is omitted in stage 1; the trainer will automatically use width=30.

Requirements
------------
pip install pyroomacoustics torchaudio soundfile numpy scipy tqdm
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torchaudio
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from tqdm import tqdm

from scipy.signal import resample_poly
import numpy as np
import soundfile as sf


# =========================
# Configuration structures
# =========================

@dataclass
class SplitConfig:
    name: str
    num_items: int  # number of trainer items = number of clean files


@dataclass
class GenConfig:
    librispeech_root: Path
    demand_root: Path
    output_root: Path
    sample_rate: int = 16000
    n_speakers: int = 3
    n_mics: int = 4
    mic_radius_m: float = 0.031
    room_w_min: float = 6.0
    room_w_max: float = 9.0
    room_d_min: float = 6.0
    room_d_max: float = 9.0
    room_h: float = 3.0
    rt60_min: float = 0.3
    rt60_max: float = 0.5
    wall_margin_m: float = 0.3
    mic_height_m: float = 1.0
    source_z_min: float = 1.0
    source_z_max: float = 1.8
    dominant_rms_db: float = -15.0
    background_rms_db_min: float = -25.0
    background_rms_db_max: float = -20.0
    source_radius_m: float = 2.0

    max_length_diff_sec: float = 5.0
    min_doa_sep_deg: float = 30.0
    ref_mic: int = 0
    random_seed: int = 42
    audio_exts: Tuple[str, ...] = (".wav", ".flac", ".mp3")


# =========================
# Utility functions
# =========================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def list_audio_files(root: Path, exts: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def read_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)

    # stereo/multi-channel -> mono
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)

    wav = wav.astype(np.float32)

    # resample if needed
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        wav = resample_poly(wav, up, down).astype(np.float32)

    return wav


def rms(sig: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(sig)) + 1e-12))


def scale_to_rms_db(sig: np.ndarray, db: float) -> np.ndarray:
    target = 10.0 ** (db / 20.0)
    current = rms(sig)
    if current < 1e-10:
        return sig.copy()
    return (sig * (target / current)).astype(np.float32)


def peak_normalize(sig: np.ndarray, peak: float = 0.95) -> np.ndarray:
    maxv = float(np.max(np.abs(sig)) + 1e-12)
    if maxv <= peak:
        return sig.astype(np.float32)
    return (sig * (peak / maxv)).astype(np.float32)


def respeaker_4mic_positions_3d(
    center_xyz: np.ndarray,
    radius: float = 0.031,
) -> np.ndarray:
    """
    ReSpeaker 4-mic physical geometry.

    Coordinate convention from Seeed image:
        DOA 0   = right  (+x)
        DOA 90  = top    (+y)
        DOA 180 = left   (-x)
        DOA 270 = bottom (-y)

    Mic physical positions:
        MIC1 = 45 deg
        MIC2 = 135 deg
        MIC3 = 225 deg
        MIC4 = 315 deg
    """
    mic_angles = [45, 135, 225, 315]

    mic_positions = np.stack(
        [
            center_xyz
            + np.array(
                [
                    radius * np.cos(np.deg2rad(a)),
                    radius * np.sin(np.deg2rad(a)),
                    0.0,
                ],
                dtype=np.float64,
            )
            for a in mic_angles
        ],
        axis=1,
    )

    return mic_positions.astype(np.float64)


def sample_room(cfg: GenConfig) -> Tuple[np.ndarray, float]:
    w = random.uniform(cfg.room_w_min, cfg.room_w_max)
    d = random.uniform(cfg.room_d_min, cfg.room_d_max)
    h = cfg.room_h
    rt60 = random.uniform(cfg.rt60_min, cfg.rt60_max)
    return np.array([w, d, h], dtype=np.float64), rt60


def sample_source_position(room_dim: np.ndarray, cfg: GenConfig) -> np.ndarray:
    m = cfg.wall_margin_m
    x = random.uniform(m, room_dim[0] - m)
    y = random.uniform(m, room_dim[1] - m)
    z = random.uniform(cfg.source_z_min, cfg.source_z_max)
    return np.array([x, y, z], dtype=np.float64)


def compute_doa_deg(src_xyz: np.ndarray, mic_center_xyz: np.ndarray) -> int:
    """
    DOA in degrees in [0, 359], based on xy-plane angle wrt mic array center.
    """
    dx = src_xyz[0] - mic_center_xyz[0]
    dy = src_xyz[1] - mic_center_xyz[1]
    ang = math.degrees(math.atan2(dy, dx)) % 360.0
    return int(round(ang)) % 360

def get_audio_duration_sec(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate

def unique_speaker_files_with_similar_length(
    files: List[Path],
    n_needed: int,
    max_diff_sec: float,
    max_attempts: int = 5000,
) -> List[Path]:
    """
    Sample files from distinct speakers whose durations differ by at most max_diff_sec.
    This avoids cropping/tile while making real overlap more likely.
    """
    by_speaker = {}

    for p in files:
        parts = p.parts
        spk = parts[-3] if len(parts) >= 3 else str(p.parent)
        by_speaker.setdefault(spk, []).append(p)

    speakers = list(by_speaker.keys())

    if len(speakers) < n_needed:
        raise ValueError(f"Need {n_needed} distinct speakers, but found only {len(speakers)}")

    for _ in range(max_attempts):
        chosen_spk = random.sample(speakers, n_needed)
        chosen_files = [random.choice(by_speaker[s]) for s in chosen_spk]

        durations = [get_audio_duration_sec(p) for p in chosen_files]

        if max(durations) - min(durations) <= max_diff_sec:
            return chosen_files

    raise RuntimeError(
        f"Could not find {n_needed} distinct-speaker clips with length difference "
        f"<= {max_diff_sec} seconds after {max_attempts} attempts."
    )


def build_room(room_dim: np.ndarray, rt60: float, fs: int) -> pra.ShoeBox:
    """
    Create a pyroomacoustics room, trying a couple of constructor styles for compatibility.
    """
    absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    # Newer versions
    try:
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order,
        )
        return room
    except Exception:
        pass

    # Older fallback
    try:
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            absorption=absorption,
            max_order=max_order,
        )
        return room
    except Exception as e:
        raise RuntimeError(f"Failed to create room with pyroomacoustics: {e}")


def simulate_multichannel(
    room_dim: np.ndarray,
    rt60: float,
    mic_positions: np.ndarray,
    source_positions: List[np.ndarray],
    source_signals: List[np.ndarray],
    fs: int,
) -> np.ndarray:
    """
    Simulate mixture for all sources. Returns [M, T].
    """
    room = build_room(room_dim, rt60, fs)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
    for pos, sig in zip(source_positions, source_signals):
        room.add_source(pos, signal=sig)
    room.simulate()
    return np.asarray(room.mic_array.signals, dtype=np.float32)


def simulate_single_target_refmic(
    room_dim: np.ndarray,
    rt60: float,
    mic_positions: np.ndarray,
    target_position: np.ndarray,
    target_signal: np.ndarray,
    fs: int,
    ref_mic: int,
) -> np.ndarray:
    """
    Simulate the target speaker alone in the same room and return only the reference mic.
    This yields a reverberant target aligned with the mixture timeline.
    """
    room = build_room(room_dim, rt60, fs)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
    room.add_source(target_position, signal=target_signal)
    room.simulate()
    target_mc = np.asarray(room.mic_array.signals, dtype=np.float32)
    return target_mc[ref_mic].astype(np.float32)


def save_wav(path: Path, sig: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), sig, sr)

def get_text_reference(audio_path: Path) -> str:
    """
    Extract transcript from LibriSpeech .trans.txt file.
    """
    # Example: 19-198-0001.flac
    file_id = audio_path.stem  # "19-198-0001"

    # Find transcript file: 19-198.trans.txt
    speaker_chapter = "-".join(file_id.split("-")[:2])  # "19-198"
    txt_path = audio_path.parent / f"{speaker_chapter}.trans.txt"

    if not txt_path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {txt_path}")

    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith(file_id):
                text = line[len(file_id):].strip()
                return text
    raise ValueError(f"Transcript for {file_id} not found in {txt_path}")


def angular_distance_deg(a: float, b: float) -> float:
    """
    Smallest circular distance between two angles in degrees.
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def sample_doas_with_min_separation(
    n_sources: int,
    min_sep_deg: float = 30.0,
) -> List[float]:
    """
    Sample DOAs directly on [0, 360) such that every pair is separated
    by at least min_sep_deg.

    This avoids failing due to random 2D position rejection.
    """
    if n_sources * min_sep_deg > 360:
        raise ValueError(
            f"Impossible to place {n_sources} sources with {min_sep_deg}° minimum separation."
        )

    # Step 1: reserve the minimum separation for each source
    # Total reserved angle = n_sources * min_sep_deg
    # Remaining free angle is distributed randomly into n_sources gaps
    free_angle = 360.0 - n_sources * min_sep_deg

    # Randomly split free_angle into n_sources nonnegative parts
    gap_fracs = np.random.dirichlet(np.ones(n_sources))
    extra_gaps = free_angle * gap_fracs

    # Build ordered angles on the circle
    start = random.uniform(0.0, 360.0)
    doas = [start]
    current = start
    for i in range(1, n_sources):
        current += min_sep_deg + extra_gaps[i - 1]
        doas.append(current % 360.0)

    random.shuffle(doas)
    return doas

def max_radius_in_room_for_angle(
    room_dim: np.ndarray,
    mic_center: np.ndarray,
    theta_deg: float,
    wall_margin_m: float,
) -> float:
    """
    Maximum radius from mic_center along angle theta_deg such that the point
    stays inside the room with wall margin.
    """
    theta = math.radians(theta_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    x_min = wall_margin_m
    x_max = room_dim[0] - wall_margin_m
    y_min = wall_margin_m
    y_max = room_dim[1] - wall_margin_m

    radii = []

    if abs(dx) > 1e-12:
        if dx > 0:
            radii.append((x_max - mic_center[0]) / dx)
        else:
            radii.append((x_min - mic_center[0]) / dx)

    if abs(dy) > 1e-12:
        if dy > 0:
            radii.append((y_max - mic_center[1]) / dy)
        else:
            radii.append((y_min - mic_center[1]) / dy)

    # keep only positive radii
    radii = [r for r in radii if r > 0]

    if not radii:
        raise RuntimeError(f"Could not compute valid radius for angle {theta_deg}")

    return min(radii)

def source_position_from_doa(
    room_dim: np.ndarray,
    mic_center: np.ndarray,
    theta_deg: float,
    cfg: GenConfig,
    fixed_radius_m: float = 2.0,
) -> np.ndarray:
    """
    Place source at a fixed distance from the microphone center.
    This keeps DOA controlled and reduces distance-induced volume variation.
    """
    r_max = max_radius_in_room_for_angle(
        room_dim=room_dim,
        mic_center=mic_center,
        theta_deg=theta_deg,
        wall_margin_m=cfg.wall_margin_m,
    )

    if fixed_radius_m > r_max:
        r = 0.8 * r_max
    else:
        r = fixed_radius_m

    theta = math.radians(theta_deg)

    x = mic_center[0] + r * math.cos(theta)
    y = mic_center[1] + r * math.sin(theta)
    z = cfg.mic_height_m   # or random height if you still want height variation

    return np.array([x, y, z], dtype=np.float64)

# =========================
# Main generation logic
# =========================

class Stage1DatasetBuilder:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        set_global_seed(cfg.random_seed)

        self.speech_files = list_audio_files(cfg.librispeech_root, cfg.audio_exts)
        self.noise_files = list_audio_files(cfg.demand_root, cfg.audio_exts)

        if not self.speech_files:
            raise FileNotFoundError(f"No speech files found under {cfg.librispeech_root}")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise files found under {cfg.demand_root}")

    def _sample_scene_audio(self) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
        """
        Sample 3 speakers with similar durations.

        Speaker 0: dominant target speaker
        Speaker 1-2: quieter background speakers

        No cropping or tiling is applied to speech clips.
        """
        cfg = self.cfg

        speech_paths = unique_speaker_files_with_similar_length(
            files=self.speech_files,
            n_needed=cfg.n_speakers,
            max_diff_sec=cfg.max_length_diff_sec,
        )

        speech_clips: List[np.ndarray] = []
        text_refs: List[str] = []

        for i, p in enumerate(speech_paths):
            sig = read_audio_mono(p, cfg.sample_rate)

            if i == 0:
                # Dominant speaker
                sig = scale_to_rms_db(sig, cfg.dominant_rms_db)
            else:
                # Background speakers
                bg_db = random.uniform(cfg.background_rms_db_min, cfg.background_rms_db_max)
                sig = scale_to_rms_db(sig, bg_db)

            speech_clips.append(sig)
            text_refs.append(get_text_reference(p))

        max_speech_len = max(len(sig) for sig in speech_clips)

        noise_path = random.choice(self.noise_files)
        noise = read_audio_mono(noise_path, cfg.sample_rate)

        # Noise can be cropped/repeated because it has no transcript.
        if len(noise) < max_speech_len:
            repeat_times = int(np.ceil(max_speech_len / len(noise)))
            noise = np.tile(noise, repeat_times)

        noise = noise[:max_speech_len]
        noise = scale_to_rms_db(noise, random.uniform(-35.0, -30.0))

        return speech_clips, noise, text_refs

    def _generate_scene(self, scene_id: int) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
        """
        Returns:
            mixture_mc: [M, T]
            target_refs: list of 6 mono reverberant target signals, one per speaker
            speaker_doas: list of integer DOAs, one per speaker
        """
        cfg = self.cfg
        room_dim, rt60 = sample_room(cfg)
        mic_center = np.array([room_dim[0] / 2.0, room_dim[1] / 2.0, cfg.mic_height_m], dtype=np.float64)
        mic_positions = respeaker_4mic_positions_3d(mic_center, radius=cfg.mic_radius_m)

        speech_clips, noise_clip, text_refs = self._sample_scene_audio()

        # Positions
        # Sample speaker DOAs first, then place speakers along those DOAs
        speaker_doas = sample_doas_with_min_separation(
            n_sources=cfg.n_speakers,
            min_sep_deg=cfg.min_doa_sep_deg,
        )

        spk_positions = [
            source_position_from_doa(
                room_dim,
                mic_center,
                doa,
                cfg,
                fixed_radius_m=cfg.source_radius_m,
            )
            for doa in speaker_doas
        ]

        # Noise remains unconstrained
        noise_position = sample_source_position(room_dim, cfg)

        # DOAs for the 6 speakers
        speaker_doas = [compute_doa_deg(p, mic_center) for p in spk_positions]

        # Full mixture = 6 speakers + 1 noise
        all_positions = spk_positions + [noise_position]
        all_signals = speech_clips + [noise_clip]
        mixture_mc = simulate_multichannel(
            room_dim=room_dim,
            rt60=rt60,
            mic_positions=mic_positions,
            source_positions=all_positions,
            source_signals=all_signals,
            fs=cfg.sample_rate,
        )

        # Per-speaker target at reference mic only
        target_refs: List[np.ndarray] = []
        for i in range(cfg.n_speakers):
            target_ref = simulate_single_target_refmic(
                room_dim=room_dim,
                rt60=rt60,
                mic_positions=mic_positions,
                target_position=spk_positions[i],
                target_signal=speech_clips[i],
                fs=cfg.sample_rate,
                ref_mic=cfg.ref_mic,
            )
            target_refs.append(target_ref)

        # Mild peak normalization per file family to avoid clipping
        # Keep relative structure intact, but protect write-out.
        mixture_mc = peak_normalize(mixture_mc, peak=0.95)
        target_refs = [peak_normalize(t, peak=0.95) for t in target_refs]

        return mixture_mc.astype(np.float32), target_refs, speaker_doas, text_refs

    def build_split(self, split: SplitConfig) -> None:
        cfg = self.cfg
        split_root = cfg.output_root / split.name
        clean_root = split_root / "clean"
        mic_root = split_root / "mic"
        text_root = split_root / "text"
        clean_root.mkdir(parents=True, exist_ok=True)
        mic_root.mkdir(parents=True, exist_ok=True)
        text_root.mkdir(parents=True, exist_ok=True)


        items_written = 0
        scene_id = 0

        pbar = tqdm(total=split.num_items, desc=f"Generating {split.name}")
        while items_written < split.num_items:
            mixture_mc, target_refs, speaker_doas, text_refs = self._generate_scene(scene_id)

            # Save only the dominant speaker voice only -----------------------------------------------------------
            for spk_idx in [0]:
                if items_written >= split.num_items:
                    break

                doa = speaker_doas[spk_idx]

                clean_name = f"clean_fileid_{scene_id}_doa{doa}_spk{spk_idx + 1}.wav"
                mic_name = f"mic_fileid_{scene_id}_doa{doa}_3spk.wav"
                text_name = f"text_fileid_{scene_id}_doa{doa}_spk{spk_idx + 1}.txt"

                clean_path = clean_root / clean_name
                mic_path = mic_root / mic_name
                text_path = text_root / text_name

                # Save text file for the target speaker
                text_path.parent.mkdir(parents=True, exist_ok=True)
                with open(text_path, "w") as f:
                    f.write(text_refs[spk_idx] + "\n")

                # Save mono clean target (trainer can repeat if needed)
                save_wav(clean_path, target_refs[spk_idx], cfg.sample_rate)

                # Save the same multichannel mixture under the DOA-specific name expected by the trainer
                # soundfile writes [T, C], while our array is [C, T]
                save_wav(mic_path, mixture_mc.T, cfg.sample_rate)

                items_written += 1
                pbar.update(1)

            scene_id += 1

        pbar.close()


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate stage-1 DSENet-style dataset.")
    parser.add_argument("--librispeech_root", type=str, default=r"D:\邵鹏远\UCL\博1\code\DSENet\data\LibriSpeech", help="Root folder of LibriSpeech")
    parser.add_argument("--demand_root", type=str, default=r"D:\邵鹏远\UCL\博1\code\DSENet\data\DEMAND", help="Root folder of DEMAND noise")
    parser.add_argument("--output_root", type=str, default=r"D:\邵鹏远\UCL\博1\code\Whisper_ASR\data\dataset_4mic_3spk_dominant", help="Output dataset root")

    parser.add_argument("--eval_items", type=int, default=1200, help="Number of ASR evaluation items")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--ref_mic", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = GenConfig(
        librispeech_root=Path(args.librispeech_root),
        demand_root=Path(args.demand_root),
        output_root=Path(args.output_root),
        sample_rate=args.sample_rate,
        random_seed=args.seed,
        ref_mic=args.ref_mic,
    )

    builder = Stage1DatasetBuilder(cfg)
    builder.build_split(SplitConfig(name="Eval", num_items=args.eval_items))


if __name__ == "__main__":
    main()