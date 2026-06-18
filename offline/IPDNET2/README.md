# Offline Python Files

This folder contains offline or offline-style evaluation scripts for the
IPDNet2 -> DSENet -> Whisper audition pipeline. The scripts generally load the
models once, process saved WAV files from `data/`, and write CSV/JSON results
under `offline/results/`.

## File Descriptions

| File | Description |
| --- | --- |
| `pipeline_full.py` | Runs the full offline SSL -> DSE -> Whisper pipeline over full-scene multichannel microphone files. It estimates DOAs with IPDNet2, matches predicted DOAs to target DOAs, enhances with DSENet, transcribes each enhanced target with Whisper, and reports per-target plus aggregate WER. |
| `pipeline_1asr.py` | Runs a scene-level one-ASR benchmark on full-scene files. For each scene it estimates three DOAs, enhances all candidates, selects the loudest enhanced signal, runs Whisper once, and scores that single transcript against dominant speaker 1 text. |
| `pipeline_1asr_4s.py` | Runs the one-ASR pipeline on 4-second chunked microphone files. It transcribes each selected chunk output, concatenates chunk hypotheses by scene, then computes scene-level WER against the dominant speaker 1 transcript. |
| `pipeline_realtime_1asr.py` | Runs a timing-focused realtime-style offline benchmark. It estimates DOAs, enhances a batch of candidate DOAs, selects the loudest enhanced signal, runs Whisper once, and records latency/RTF-style timing plus DOA error statistics. It does not require ground-truth text. NO WER|
| `eval_doa_wer.py` | Small analysis script that loads a WER details CSV, filters rows by a DOA error threshold, and prints sample counts plus mean/corpus WER for the filtered subset. |
| `plot_doa_error_dist.py` | Plots the distribution of `selected_doa_error_deg` from a results CSV using a histogram and KDE curve, then saves the figure as `selected_doa_error_hist_kde_0_180.png`. |
| `total_audio_length.py` | Utility script that scans a hard-coded microphone WAV folder, sums file durations with `soundfile`, and prints total file count, seconds, minutes, and hours. |

## Common Inputs

- Model checkpoints are expected under `Models/SSL` and `Models/DSE`.
- Most pipeline scripts read microphone, clean, and text data from dataset
  folders under `data/`, with each script choosing the full-scene, 4-second, or
  4-second-full dataset variant.
- The pipeline scripts expose CLI arguments for the main paths, model labels,
  device choice, DOA/VAD settings, and optional enhanced WAV saving.

## Common Outputs

- Detailed per-item CSV files are written to `offline/results/`.
- Summary JSON files contain aggregate WER, DOA error, timing, and realtime
  metrics, depending on the script.
- When `--save_enhanced` is used, selected enhanced WAV files are saved under a
  results subfolder.
