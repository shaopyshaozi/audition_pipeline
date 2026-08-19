#!/usr/bin/env python3
"""
Create a controlled 3-speaker DoA-separation evaluation set.

Experiment design
-----------------
Each acoustic scene contains:
    spk1: target speaker at theta
    spk2: nearest interferer at theta + doa_sep
    spk3: far interferer at theta + 180
    noise: independent DEMAND noise at a random room position

The independent variable is target-to-nearest-interferer DoA separation:
    10, 20, 30, 50 degrees by default.

All three speakers are scaled to near-equal RMS levels sampled from
    [-23, -18] dB by default.

All three speakers are saved as target items. Each speaker has its own
clean reference and transcript, while the same 3-speaker mixture is saved
under that speaker's target DoA.

Output layout
-------------
dataset_root/
    Eval/
        clean/
        mic/
        text/
        metadata.csv

Filename convention:
    clean_fileid_<sceneid>_doa<target_doa>_sep<doa_sep>_spk<k>.wav
    mic_fileid_<sceneid>_doa<target_doa>_sep<doa_sep>_3spk.wav
    text_fileid_<sceneid>_doa<target_doa>_sep<doa_sep>_spk<k>.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from data_creation_4mics import (
    GenConfig,
    angular_distance_deg,
    compute_doa_deg,
    get_text_reference,
    list_audio_files,
    peak_normalize,
    read_audio_mono,
    respeaker_4mic_positions_3d,
    sample_room,
    sample_source_position,
    save_wav,
    scale_to_rms_db,
    set_global_seed,
    simulate_multichannel,
    simulate_single_target_refmic,
    source_position_from_doa,
    unique_speaker_files_with_similar_length,
)


@dataclass
class SeparationSplitConfig:
    name: str
    items_per_sep: int


@dataclass
class SeparationScene:
    mixture_mc: np.ndarray
    target_refs: List[np.ndarray]
    text_refs: List[str]
    metadata: Dict[str, object]


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list, e.g. 10,20,30,50.")
    return items


def signed_nearest_interferer_doa(target_doa: float, sep_deg: int, sign: int) -> float:
    return (target_doa + sign * abs(sep_deg)) % 360.0


def far_interferer_doa(target_doa: float) -> float:
    return (target_doa + 180.0) % 360.0


def circular_mean_deg(values: Sequence[float]) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64))
    mean_angle = math.degrees(math.atan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    return mean_angle % 360.0


class DoASeparationDatasetBuilder:
    def __init__(
        self,
        cfg: GenConfig,
        doa_separations: Sequence[int],
        randomize_nearest_side: bool,
        speaker_rms_db_min: float,
        speaker_rms_db_max: float,
    ):
        if cfg.n_speakers != 3:
            raise ValueError("This controlled experiment requires exactly 3 speakers.")
        if speaker_rms_db_min > speaker_rms_db_max:
            raise ValueError("speaker_rms_db_min must be <= speaker_rms_db_max.")

        self.cfg = cfg
        self.doa_separations = [int(abs(v)) for v in doa_separations]
        self.randomize_nearest_side = randomize_nearest_side
        self.speaker_rms_db_min = speaker_rms_db_min
        self.speaker_rms_db_max = speaker_rms_db_max
        set_global_seed(cfg.random_seed)

        self.speech_files = list_audio_files(cfg.librispeech_root, cfg.audio_exts)
        self.noise_files = list_audio_files(cfg.demand_root, cfg.audio_exts)

        if not self.speech_files:
            raise FileNotFoundError(f"No speech files found under {cfg.librispeech_root}")
        if not self.noise_files:
            raise FileNotFoundError(f"No noise files found under {cfg.demand_root}")

    def _sample_scene_audio(self) -> Tuple[List[np.ndarray], np.ndarray, List[str], List[str], List[float]]:
        cfg = self.cfg
        speech_paths = unique_speaker_files_with_similar_length(
            files=self.speech_files,
            n_needed=cfg.n_speakers,
            max_diff_sec=cfg.max_length_diff_sec,
        )

        speech_clips: List[np.ndarray] = []
        text_refs: List[str] = []
        speech_source_files: List[str] = []
        speaker_rms_dbs: List[float] = []

        for path in speech_paths:
            sig = read_audio_mono(path, cfg.sample_rate)
            speaker_db = random.uniform(self.speaker_rms_db_min, self.speaker_rms_db_max)
            sig = scale_to_rms_db(sig, speaker_db)

            speech_clips.append(sig)
            text_refs.append(get_text_reference(path))
            speech_source_files.append(str(path))
            speaker_rms_dbs.append(speaker_db)

        max_speech_len = max(len(sig) for sig in speech_clips)
        noise_path = random.choice(self.noise_files)
        noise = read_audio_mono(noise_path, cfg.sample_rate)

        if len(noise) < max_speech_len:
            repeat_times = int(np.ceil(max_speech_len / len(noise)))
            noise = np.tile(noise, repeat_times)

        noise = noise[:max_speech_len]
        noise = scale_to_rms_db(noise, random.uniform(-35.0, -30.0))

        return speech_clips, noise, text_refs, speech_source_files, speaker_rms_dbs

    def _nearest_sign(self) -> int:
        if self.randomize_nearest_side:
            return -1 if random.random() < 0.5 else 1
        return 1

    def _generate_scene(self, scene_id: int, doa_sep: int) -> SeparationScene:
        cfg = self.cfg
        room_dim, rt60 = sample_room(cfg)
        mic_center = np.array(
            [room_dim[0] / 2.0, room_dim[1] / 2.0, cfg.mic_height_m],
            dtype=np.float64,
        )
        mic_positions = respeaker_4mic_positions_3d(mic_center, radius=cfg.mic_radius_m)

        speech_clips, noise_clip, text_refs, speech_source_files, speaker_rms_dbs = self._sample_scene_audio()

        target_doa_requested = random.uniform(0.0, 360.0)
        nearest_sign = self._nearest_sign()
        spk2_doa_requested = signed_nearest_interferer_doa(
            target_doa_requested,
            doa_sep,
            nearest_sign,
        )
        spk3_doa_requested = far_interferer_doa(target_doa_requested)
        requested_doas = [target_doa_requested, spk2_doa_requested, spk3_doa_requested]

        spk_positions = [
            source_position_from_doa(
                room_dim,
                mic_center,
                doa,
                cfg,
                fixed_radius_m=cfg.source_radius_m,
            )
            for doa in requested_doas
        ]

        noise_position = sample_source_position(room_dim, cfg)
        speaker_doas = [compute_doa_deg(position, mic_center) for position in spk_positions]

        target_to_spk2_sep = angular_distance_deg(speaker_doas[0], speaker_doas[1])
        target_to_spk3_sep = angular_distance_deg(speaker_doas[0], speaker_doas[2])
        nearest_target_interferer_sep = min(target_to_spk2_sep, target_to_spk3_sep)
        mean_target_interferer_sep = float(np.mean([target_to_spk2_sep, target_to_spk3_sep]))
        min_pairwise_sep = min(
            angular_distance_deg(speaker_doas[0], speaker_doas[1]),
            angular_distance_deg(speaker_doas[0], speaker_doas[2]),
            angular_distance_deg(speaker_doas[1], speaker_doas[2]),
        )

        mixture_mc = simulate_multichannel(
            room_dim=room_dim,
            rt60=rt60,
            mic_positions=mic_positions,
            source_positions=spk_positions + [noise_position],
            source_signals=speech_clips + [noise_clip],
            fs=cfg.sample_rate,
        )

        target_refs: List[np.ndarray] = []
        for spk_idx in range(cfg.n_speakers):
            target_ref = simulate_single_target_refmic(
                room_dim=room_dim,
                rt60=rt60,
                mic_positions=mic_positions,
                target_position=spk_positions[spk_idx],
                target_signal=speech_clips[spk_idx],
                fs=cfg.sample_rate,
                ref_mic=cfg.ref_mic,
            )
            target_refs.append(target_ref)

        mixture_mc = peak_normalize(mixture_mc, peak=0.95).astype(np.float32)
        target_refs = [peak_normalize(target_ref, peak=0.95).astype(np.float32) for target_ref in target_refs]

        metadata: Dict[str, object] = {
            "scene_id": scene_id,
            "condition_doa_sep_deg": int(doa_sep),
            "nearest_target_interferer_sep_deg": round(float(nearest_target_interferer_sep), 3),
            "target_to_spk2_sep_deg": round(float(target_to_spk2_sep), 3),
            "target_to_spk3_sep_deg": round(float(target_to_spk3_sep), 3),
            "mean_target_interferer_sep_deg": round(mean_target_interferer_sep, 3),
            "min_pairwise_sep_deg": round(float(min_pairwise_sep), 3),
            "nearest_interferer_sign": nearest_sign,
            "target_doa_requested": round(float(target_doa_requested), 3),
            "spk2_doa_requested": round(float(spk2_doa_requested), 3),
            "spk3_doa_requested": round(float(spk3_doa_requested), 3),
            "target_doa": int(speaker_doas[0]),
            "spk1_doa": int(speaker_doas[0]),
            "spk2_doa": int(speaker_doas[1]),
            "spk3_doa": int(speaker_doas[2]),
            "doa_circular_mean": round(circular_mean_deg(speaker_doas), 3),
            "room_w": round(float(room_dim[0]), 4),
            "room_d": round(float(room_dim[1]), 4),
            "room_h": round(float(room_dim[2]), 4),
            "rt60": round(float(rt60), 4),
            "source_radius_m": round(float(cfg.source_radius_m), 4),
            "mic_center_x": round(float(mic_center[0]), 4),
            "mic_center_y": round(float(mic_center[1]), 4),
            "mic_center_z": round(float(mic_center[2]), 4),
            "spk1_source_file": speech_source_files[0],
            "spk2_source_file": speech_source_files[1],
            "spk3_source_file": speech_source_files[2],
            "spk1_rms_db": round(float(speaker_rms_dbs[0]), 3),
            "spk2_rms_db": round(float(speaker_rms_dbs[1]), 3),
            "spk3_rms_db": round(float(speaker_rms_dbs[2]), 3),
        }

        return SeparationScene(
            mixture_mc=mixture_mc,
            target_refs=target_refs,
            text_refs=text_refs,
            metadata=metadata,
        )

    def build_split(self, split: SeparationSplitConfig) -> None:
        cfg = self.cfg
        split_root = cfg.output_root / split.name
        clean_root = split_root / "clean"
        mic_root = split_root / "mic"
        text_root = split_root / "text"
        clean_root.mkdir(parents=True, exist_ok=True)
        mic_root.mkdir(parents=True, exist_ok=True)
        text_root.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        total_scenes = split.items_per_sep * len(self.doa_separations)
        pbar = tqdm(total=total_scenes, desc=f"Generating {split.name} scenes")

        scene_id = 0
        for doa_sep in self.doa_separations:
            for _ in range(split.items_per_sep):
                scene = self._generate_scene(scene_id=scene_id, doa_sep=doa_sep)

                for spk_idx in range(cfg.n_speakers):
                    speaker_id = spk_idx + 1
                    target_doa = int(scene.metadata[f"spk{speaker_id}_doa"])

                    clean_name = f"clean_fileid_{scene_id}_doa{target_doa}_sep{doa_sep}_spk{speaker_id}.wav"
                    mic_name = f"mic_fileid_{scene_id}_doa{target_doa}_sep{doa_sep}_3spk.wav"
                    text_name = f"text_fileid_{scene_id}_doa{target_doa}_sep{doa_sep}_spk{speaker_id}.txt"

                    clean_path = clean_root / clean_name
                    mic_path = mic_root / mic_name
                    text_path = text_root / text_name

                    save_wav(clean_path, scene.target_refs[spk_idx], cfg.sample_rate)
                    save_wav(mic_path, scene.mixture_mc.T, cfg.sample_rate)
                    text_path.write_text(scene.text_refs[spk_idx] + "\n", encoding="utf-8")

                    row = dict(scene.metadata)
                    row.update(
                        {
                            "split": split.name,
                            "target_speaker_id": speaker_id,
                            "target_item_doa": target_doa,
                            "target_source_file": scene.metadata[f"spk{speaker_id}_source_file"],
                            "clean_file": str(clean_path),
                            "mic_file": str(mic_path),
                            "text_file": str(text_path),
                        }
                    )
                    rows.append(row)

                scene_id += 1
                pbar.update(1)

        pbar.close()
        self._write_metadata(split_root / "metadata.csv", rows)
        self._write_config(split_root / "generation_config.json", split, rows)

    def _write_metadata(self, path: Path, rows: Sequence[Dict[str, object]]) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_config(
        self,
        path: Path,
        split: SeparationSplitConfig,
        rows: Sequence[Dict[str, object]],
    ) -> None:
        cfg = self.cfg
        payload = {
            "experiment": "3-speaker target-nearest-interferer DoA separation test",
            "split": split.name,
            "items_per_sep": split.items_per_sep,
            "scenes_per_sep": split.items_per_sep,
            "total_scenes": split.items_per_sep * len(self.doa_separations),
            "total_target_items": len(rows),
            "doa_separations": self.doa_separations,
            "layout": {
                "spk1": "target at theta",
                "spk2": "nearest interferer at theta + condition_doa_sep_deg",
                "spk3": "far interferer at theta + 180 deg",
            },
            "randomize_nearest_side": self.randomize_nearest_side,
            "sample_rate": cfg.sample_rate,
            "n_speakers": cfg.n_speakers,
            "n_mics": cfg.n_mics,
            "source_radius_m": cfg.source_radius_m,
            "speaker_rms_db_min": self.speaker_rms_db_min,
            "speaker_rms_db_max": self.speaker_rms_db_max,
            "rt60_min": cfg.rt60_min,
            "rt60_max": cfg.rt60_max,
            "random_seed": cfg.random_seed,
            "ref_mic": cfg.ref_mic,
            "librispeech_root": str(cfg.librispeech_root),
            "demand_root": str(cfg.demand_root),
            "output_root": str(cfg.output_root),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_data_path(windows_path: str, wsl_path: str) -> Path:
    return Path(windows_path if os.name == "nt" else wsl_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 3-speaker DoA separation stress-test dataset."
    )
    parser.add_argument(
        "--librispeech_root",
        type=Path,
        default=default_data_path(
            r"D:\邵鹏远\UCL\博1\code\DSENet\data\LibriSpeech",
            "/mnt/d/邵鹏远/UCL/博1/code/DSENet/data/LibriSpeech",
        ),
    )
    parser.add_argument(
        "--demand_root",
        type=Path,
        default=default_data_path(
            r"D:\邵鹏远\UCL\博1\code\DSENet\data\DEMAND",
            "/mnt/d/邵鹏远/UCL/博1/code/DSENet/data/DEMAND",
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=default_data_path(
            r"D:\邵鹏远\UCL\博1\code\audition_pipeline\data\dataset_doa_sep_3spk",
            "/mnt/d/邵鹏远/UCL/博1/code/audition_pipeline/data/dataset_doa_sep_3spk",
        ),
    )
    parser.add_argument(
        "--doa_separations",
        type=parse_int_list,
        default=parse_int_list("5,10,15,20,30"),
        help="Comma-separated target-to-nearest-interferer separations.",
    )
    parser.add_argument("--items_per_sep", type=int, default=100)
    parser.add_argument("--split_name", type=str, default="Eval")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_mic", type=int, default=0)
    parser.add_argument("--source_radius_m", type=float, default=2.0)
    parser.add_argument("--speaker_rms_db_min", type=float, default=-22.0)
    parser.add_argument("--speaker_rms_db_max", type=float, default=-18.0)
    parser.add_argument(
        "--randomize_nearest_side",
        action="store_true",
        help="Randomly use theta + separation or theta - separation for spk2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = GenConfig(
        librispeech_root=args.librispeech_root,
        demand_root=args.demand_root,
        output_root=args.output_root,
        sample_rate=args.sample_rate,
        n_speakers=3,
        source_radius_m=args.source_radius_m,
        random_seed=args.seed,
        ref_mic=args.ref_mic,
    )

    builder = DoASeparationDatasetBuilder(
        cfg=cfg,
        doa_separations=args.doa_separations,
        randomize_nearest_side=args.randomize_nearest_side,
        speaker_rms_db_min=args.speaker_rms_db_min,
        speaker_rms_db_max=args.speaker_rms_db_max,
    )
    builder.build_split(
        SeparationSplitConfig(
            name=args.split_name,
            items_per_sep=args.items_per_sep,
        )
    )


if __name__ == "__main__":
    main()
