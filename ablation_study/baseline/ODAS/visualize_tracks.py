#!/usr/bin/env python3
"""Visualize ODAS ``sst.tracked`` JSON output.

ODAS writes consecutive JSON objects rather than one JSON array.  This script
parses that format, keeps dynamic tracks, and writes a two-panel PNG showing
each track's azimuth and activity over time.

Example (WSL):
    python3 visualize_tracks.py \
        sep_audios/tracks_fileid_0_doa73_3spk.json \
        --gt-doas 73 169 206

The image is saved next to the input by default, so it also works in WSL
without an X server.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import matplotlib

# Saving a PNG should work in WSL without an X server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def iter_odas_frames(path: Path) -> Iterator[dict]:
    """Yield the consecutive top-level JSON objects written by ODAS."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    position = 0

    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            return

        try:
            frame, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Could not parse an ODAS JSON object near character {position}: {error}"
            ) from error

        if not isinstance(frame, dict):
            raise ValueError("Expected each ODAS record to be a JSON object.")
        yield frame


def circular_mean_deg(angles: list[float]) -> float:
    """Return the mean azimuth while correctly handling the 0°/360° boundary."""
    radians = [math.radians(angle) for angle in angles]
    mean = math.degrees(math.atan2(sum(map(math.sin, radians)), sum(map(math.cos, radians))))
    return mean % 360.0


def read_tracks(
    path: Path,
    sample_rate: int,
    hop_size: int,
) -> dict[int, list[tuple[float, float, float]]]:
    """Return track_id -> [(time_seconds, azimuth_degrees, activity), ...]."""
    tracks: dict[int, list[tuple[float, float, float]]] = defaultdict(list)

    for frame in iter_odas_frames(path):
        timestamp = frame.get("timeStamp")
        if not isinstance(timestamp, (int, float)):
            continue
        time_seconds = float(timestamp) * hop_size / sample_rate

        for source in frame.get("src", []):
            if source.get("tag") != "dynamic":
                continue

            try:
                track_id = int(source["id"])
                x = float(source["x"])
                y = float(source["y"])
                activity = float(source.get("activity", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

            azimuth = math.degrees(math.atan2(y, x)) % 360.0
            tracks[track_id].append((time_seconds, azimuth, activity))

    return dict(tracks)


def plot_tracks(
    tracks: dict[int, list[tuple[float, float, float]]],
    output_path: Path,
    gt_doas: list[float],
) -> None:
    if not tracks:
        raise ValueError("No dynamic ODAS tracks were found in this file.")

    figure, (azimuth_axis, activity_axis) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(12, 7),
        constrained_layout=True,
    )
    colors = plt.get_cmap("tab10")

    for color_index, track_id in enumerate(sorted(tracks)):
        samples = tracks[track_id]
        times = [sample[0] for sample in samples]
        azimuths = [sample[1] for sample in samples]
        activities = [sample[2] for sample in samples]
        color = colors(color_index)
        marker_colors = [(*color[:3], 0.12 + 0.88 * activity) for activity in activities]
        label = f"Track {track_id} (mean {circular_mean_deg(azimuths):.1f}°)"

        azimuth_axis.plot(times, azimuths, color=color, linewidth=1.2, label=label)
        azimuth_axis.scatter(
            times,
            azimuths,
            c=marker_colors,
            s=[8 + 18 * activity for activity in activities],
            linewidths=0,
        )
        activity_axis.plot(times, activities, color=color, linewidth=1.0, label=f"Track {track_id}")

    for doa in gt_doas:
        azimuth_axis.axhline(doa % 360.0, color="0.45", linestyle="--", linewidth=0.8)
        azimuth_axis.text(
            1.0,
            doa % 360.0,
            f" GT {doa:g}°",
            color="0.35",
            fontsize=9,
            va="bottom",
            transform=azimuth_axis.get_yaxis_transform(),
        )

    azimuth_axis.set_title("ODAS tracked-source azimuth")
    azimuth_axis.set_ylabel("Azimuth (degrees)")
    azimuth_axis.set_ylim(0, 360)
    azimuth_axis.set_yticks(range(0, 361, 45))
    azimuth_axis.grid(axis="y", alpha=0.25)
    azimuth_axis.legend(loc="upper right")

    activity_axis.set_title("ODAS source activity")
    activity_axis.set_xlabel("Time (seconds)")
    activity_axis.set_ylabel("Activity")
    activity_axis.set_ylim(-0.03, 1.03)
    activity_axis.grid(alpha=0.25)
    activity_axis.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved plot: {output_path}")

    for track_id in sorted(tracks):
        samples = tracks[track_id]
        azimuths = [sample[1] for sample in samples]
        activities = [sample[2] for sample in samples]
        print(
            f"Track {track_id}: {samples[0][0]:.3f}–{samples[-1][0]:.3f} s, "
            f"mean azimuth {circular_mean_deg(azimuths):.1f}°, "
            f"mean activity {sum(activities) / len(activities):.3f}"
        )

    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ODAS tracked-source JSON output.")
    parser.add_argument("tracks_json", type=Path, help="Path to tracks_*.json written by ODAS.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: next to the input JSON).",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-size", type=int, default=128)
    parser.add_argument(
        "--gt-doas",
        type=float,
        nargs="*",
        default=[],
        metavar="DOA",
        help="Optional ground-truth DOAs in degrees, drawn as dashed reference lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tracks_json.is_file():
        raise FileNotFoundError(f"Track JSON was not found: {args.tracks_json}")
    if args.sample_rate <= 0 or args.hop_size <= 0:
        raise ValueError("--sample-rate and --hop-size must both be positive.")

    output_path = args.output or args.tracks_json.with_name(f"{args.tracks_json.stem}_plot.png")
    tracks = read_tracks(args.tracks_json, args.sample_rate, args.hop_size)
    plot_tracks(tracks, output_path, args.gt_doas)


if __name__ == "__main__":
    main()
