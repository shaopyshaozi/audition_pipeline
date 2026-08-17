"""
Run one file through official PyHARK:

    AudioStreamFromMemory -> MultiFFT -> LocalizeMUSIC -> SourceTracker
        -> SourceIntervalExtender -> GHDSS -> Synthesize -> SaveWavePCM

This file intentionally imports PyHARK only here, so HARK.py can still be
imported and used for evaluation on machines that do not have PyHARK installed.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

import hark
import hark.base
import hark.core
import hark.node


CONFIG = {}


class HARKLocalization(hark.NetworkDef):
    def build(self, network: hark.Network, input: hark.DataSourceMap, output: hark.DataSinkMap):
        node_cm_identity_matrix = network.create(
            hark.node.CMIdentityMatrix,
            dispatch=hark.RepeatDispatcher,
        )
        node_operation_flag = network.create(
            hark.node.Constant,
            dispatch=hark.RepeatDispatcher,
        )
        node_localize_music = network.create(hark.node.LocalizeMUSIC)
        node_source_tracker = network.create(hark.node.SourceTracker)
        node_source_interval_extender = network.create(hark.node.SourceIntervalExtender)

        nodes = [
            node_cm_identity_matrix
                .add_input("NB_CHANNELS", CONFIG["channel_count"])
                .add_input("LENGTH", CONFIG["frame_length"]),
            node_operation_flag
                .add_input("VALUE", True),
            node_localize_music
                .add_input("INPUT", input["SPEC"])
                .add_input("NOISECM", node_cm_identity_matrix["OUTPUT"])
                .add_input("OPERATION_FLAG", node_operation_flag["OUTPUT"])
                .add_input("MUSIC_ALGORITHM", CONFIG["music_algorithm"])
                .add_input("TF_INPUT_TYPE", "FILE")
                .add_input("A_MATRIX", CONFIG["localization_tf"])
                .add_input("LENGTH", CONFIG["frame_length"])
                .add_input("SAMPLING_RATE", CONFIG["sample_rate"])
                .add_input("NUM_SOURCE", CONFIG["num_sources"])
                .add_input("MIN_DEG", CONFIG["min_deg"])
                .add_input("MAX_DEG", CONFIG["max_deg"])
                .add_input("WINDOW", CONFIG["music_window"])
                .add_input("WINDOW_TYPE", CONFIG["music_window_type"])
                .add_input("PERIOD", CONFIG["music_period"])
                .add_input("LOWER_BOUND_FREQUENCY", CONFIG["music_min_freq"])
                .add_input("UPPER_BOUND_FREQUENCY", CONFIG["music_max_freq"])
                .add_input("SPECTRUM_WEIGHT_TYPE", CONFIG["spectrum_weight_type"])
                .add_input("ENABLE_EIGENVALUE_WEIGHT", False)
                .add_input("ENABLE_OUTPUT_SPECTRUM", False)
                .add_input("DEBUG", CONFIG["debug"]),
            node_source_tracker
                .add_input("INPUT", node_localize_music["OUTPUT"])
                .add_input("THRESH", CONFIG["tracker_thresh"])
                .add_input("PAUSE_LENGTH", CONFIG["tracker_pause_length"])
                .add_input("MIN_SRC_INTERVAL", CONFIG["tracker_min_src_interval"])
                .add_input("MIN_ID", 0)
                .add_input("DEBUG", CONFIG["debug"]),
            node_source_interval_extender
                .add_input("SOURCES", node_source_tracker["OUTPUT"])
                .add_input("PREROLL_LENGTH", CONFIG["preroll_length"]),
        ]
        output.add_input("OUTPUT", node_source_interval_extender["OUTPUT"])
        return nodes


class HARKSeparation(hark.NetworkDef):
    def build(self, network: hark.Network, input: hark.DataSourceMap, output: hark.DataSinkMap):
        node_ghdss = network.create(hark.node.GHDSS)
        node_synthesize = network.create(hark.node.Synthesize)
        node_save_wave_pcm = network.create(hark.node.SaveWavePCM)

        nodes = [
            node_ghdss
                .add_input("INPUT_FRAMES", input["SPEC"])
                .add_input("INPUT_SOURCES", input["SRC_INFO"])
                .add_input("LENGTH", CONFIG["frame_length"])
                .add_input("ADVANCE", CONFIG["frame_shift"])
                .add_input("SAMPLING_RATE", CONFIG["sample_rate"])
                .add_input("TF_INPUT_TYPE", "FILE")
                .add_input("TF_CONJ_FILENAME", CONFIG["separation_tf"])
                .add_input("LC_CONST", CONFIG["ghdss_lc_const"])
                .add_input("UPDATE_METHOD_W", CONFIG["ghdss_update_method_w"]),
            node_synthesize
                .add_input("INPUT", node_ghdss["OUTPUT"])
                .add_input("LENGTH", CONFIG["frame_length"])
                .add_input("ADVANCE", CONFIG["frame_shift"])
                .add_input("SAMPLING_RATE", CONFIG["sample_rate"]),
            node_save_wave_pcm
                .add_input("INPUT", node_synthesize["OUTPUT"])
                .add_input("BASENAME", CONFIG["output_prefix"])
                .add_input("ADVANCE", CONFIG["frame_shift"])
                .add_input("SAMPLING_RATE", CONFIG["sample_rate"])
                .add_input("BITS", CONFIG["output_bits"]),
        ]
        output.add_input("OUTPUT", node_ghdss["OUTPUT"])
        return nodes


class HARKMain(hark.NetworkDef):
    def build(self, network: hark.Network, input: hark.DataSourceMap, output: hark.DataSinkMap):
        node_publisher = network.create(
            hark.node.PublishData,
            dispatch=hark.RepeatDispatcher,
            name="Publisher",
        )
        node_subscriber = network.create(
            hark.node.SubscribeData,
            name="Subscriber",
        )
        node_audio_stream = network.create(
            hark.node.AudioStreamFromMemory,
            dispatch=hark.TriggeredMultiShotDispatcher,
            name="AudioStreamFromMemory",
        )
        node_multi_fft = network.create(hark.node.MultiFFT)
        node_localization = network.create(HARKLocalization, name="HARKLocalization")
        node_separation = network.create(HARKSeparation, name="HARKSeparation")

        nodes = [
            node_audio_stream
                .add_input("INPUT", node_publisher["OUTPUT"])
                .add_input("CHANNEL_COUNT", CONFIG["channel_count"])
                .add_input("LENGTH", CONFIG["frame_length"])
                .add_input("ADVANCE", CONFIG["frame_shift"]),
            node_multi_fft
                .add_input("INPUT", node_audio_stream["AUDIO"])
                .add_input("LENGTH", CONFIG["frame_length"])
                .add_input("ADVANCE", CONFIG["frame_shift"]),
            node_localization
                .add_input("SPEC", node_multi_fft["OUTPUT"]),
            node_separation
                .add_input("SPEC", node_multi_fft["OUTPUT"])
                .add_input("SRC_INFO", node_localization["OUTPUT"]),
            node_subscriber
                .add_input("INPUT", node_separation["OUTPUT"]),
        ]
        return nodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one wav through official PyHARK LocalizeMUSIC + GHDSS.")
    parser.add_argument("--input_wav", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument("--localization_tf", type=Path, required=True)
    parser.add_argument("--separation_tf", type=Path, default=None)
    parser.add_argument("--channel_count", type=int, default=4)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_sources", type=int, default=3)
    parser.add_argument("--frame_length", type=int, default=512)
    parser.add_argument("--frame_shift", type=int, default=160)
    parser.add_argument("--music_algorithm", type=str, default="SEVD")
    parser.add_argument("--music_min_freq", type=int, default=500)
    parser.add_argument("--music_max_freq", type=int, default=2800)
    parser.add_argument("--music_window", type=int, default=50)
    parser.add_argument("--music_window_type", type=str, default="MIDDLE")
    parser.add_argument("--music_period", type=int, default=50)
    parser.add_argument("--spectrum_weight_type", type=str, default="A_Characteristic")
    parser.add_argument("--min_deg", type=int, default=-180)
    parser.add_argument("--max_deg", type=int, default=180)
    parser.add_argument("--tracker_thresh", type=float, default=25.0)
    parser.add_argument("--tracker_pause_length", type=float, default=1200.0)
    parser.add_argument("--tracker_min_src_interval", type=float, default=20.0)
    parser.add_argument("--preroll_length", type=int, default=80)
    parser.add_argument("--ghdss_lc_const", type=str, default="DIAG")
    parser.add_argument("--ghdss_update_method_w", type=str, default="ID")
    parser.add_argument("--output_bits", type=str, default="int16")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def make_frames(audio: np.ndarray, frame_shift: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[0] < frame_shift:
        pad = np.zeros((frame_shift - audio.shape[0], audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=0)
    remainder = audio.shape[0] % frame_shift
    if remainder:
        pad = np.zeros((frame_shift - remainder, audio.shape[1]), dtype=audio.dtype)
        audio = np.concatenate([audio, pad], axis=0)
    return np.lib.stride_tricks.sliding_window_view(audio, frame_shift, axis=0)[::frame_shift, :, :]


def main() -> None:
    args = parse_args()
    separation_tf = args.separation_tf or args.localization_tf
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    CONFIG.update(
        {
            "input_wav": str(args.input_wav),
            "output_prefix": str(args.output_prefix),
            "localization_tf": str(args.localization_tf),
            "separation_tf": str(separation_tf),
            "channel_count": args.channel_count,
            "sample_rate": args.sample_rate,
            "num_sources": args.num_sources,
            "frame_length": args.frame_length,
            "frame_shift": args.frame_shift,
            "music_algorithm": args.music_algorithm,
            "music_min_freq": args.music_min_freq,
            "music_max_freq": args.music_max_freq,
            "music_window": args.music_window,
            "music_window_type": args.music_window_type,
            "music_period": args.music_period,
            "spectrum_weight_type": args.spectrum_weight_type,
            "min_deg": args.min_deg,
            "max_deg": args.max_deg,
            "tracker_thresh": args.tracker_thresh,
            "tracker_pause_length": args.tracker_pause_length,
            "tracker_min_src_interval": args.tracker_min_src_interval,
            "preroll_length": args.preroll_length,
            "ghdss_lc_const": args.ghdss_lc_const,
            "ghdss_update_method_w": args.ghdss_update_method_w,
            "output_bits": args.output_bits,
            "debug": args.debug,
        }
    )

    audio, rate = sf.read(str(args.input_wav), dtype=np.int16, always_2d=True)
    if rate != args.sample_rate:
        raise ValueError(f"Expected {args.sample_rate} Hz input, got {rate} Hz: {args.input_wav}")
    if audio.shape[1] < args.channel_count:
        raise ValueError(f"Expected at least {args.channel_count} channels, got {audio.shape[1]}")
    audio = audio[:, : args.channel_count]
    frames = make_frames(audio, args.frame_shift)

    network = hark.Network.from_networkdef(HARKMain, name="HARKMain")
    publisher = network.query_nodedef("Publisher")
    subscriber = network.query_nodedef("Subscriber")
    subscriber.receive = lambda data: None

    thread = threading.Thread(target=network.execute)
    thread.start()
    try:
        for frame in frames:
            if not thread.is_alive():
                break
            publisher.push(frame)
    finally:
        publisher.close()
        if thread.ident is not None:
            thread.join()


if __name__ == "__main__":
    main()
