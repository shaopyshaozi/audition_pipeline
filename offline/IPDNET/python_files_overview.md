# Offline IPDNET Python Files Overview

This folder contains offline evaluation and analysis scripts for the saved-WAV
audition pipeline:

```text
Saved 4-mic mixtures -> IPDNET/SSL -> DSENet -> Whisper/ASR -> WER and timing analysis
```

Unlike the online Respeaker evaluation scripts, these files mostly work on
synthetic or pre-generated dataset folders under `data/`, rather than live
streaming input. The core purpose is to test how well the IPDNET -> DSENet ->
Whisper chain works when the audio is already available on disk, and to separate
three related questions:

```text
1. Does IPDNET predict useful DOAs?
2. Does DSENet enhance the selected target speaker from those DOAs?
3. Does the enhanced audio improve Whisper WER under offline evaluation?
```

Most pipeline scripts load IPDNET, DSENet, and Whisper once, then iterate over
saved multichannel WAV files. Results are written under `--out_dir`, which
defaults to:

```text
offline/IPDNET/results/<script_name>
```

Common output files include:

```text
pipeline_realtime_enhanced/
  selected enhanced WAV files, when --save_enhanced is used

pipeline_whisper_<model>_wer_details*.csv
  per-item WER, DOA, transcript, and timing details

pipeline_whisper_<model>_wer_summary*.json
  aggregate WER, DOA, and timing metrics

pipeline_realtime_<model>_details*.csv
  timing-oriented per-scene metrics

pipeline_realtime_<model>_summary*.json
  aggregate realtime factor and latency metrics
```

## Common Assumptions

Most scripts expect microphone WAV names that contain a `fileid` and one or more
DOA values, for example:

```text
mic_fileid_<id>_doa<angle>_...wav
```

The pipeline groups files by `fileid`, chooses one representative multichannel
mixture for SSL, predicts up to three DOAs, enhances DSENet candidates for those
DOAs, then either transcribes every relevant target or selects one enhanced
stream for ASR.

The main checkpoints are:

```text
Models/SSL/IPDNET/last-v1.ckpt
Models/DSE/last.ckpt
```

The main dataset roots are:

```text
data/dataset_4mic_3spk
data/dataset_4mic_3spk_4s
data/dataset_4mic_3spk_4s_full
```

## `pipeline_full.py`

This is the broad offline IPDNET -> DSENet -> Whisper evaluation script.

What it does:

- Loads IPDNET, DSENet, and Whisper once.
- Processes saved multichannel WAV files as they are found on disk.
- Uses IPDNET-predicted DOAs, not filename ground-truth DOAs, as DSENet inputs.
- For each fileid, matches target text files and target DOAs.
- Chooses the nearest predicted DOA for each target speaker.
- Runs Whisper on the corresponding enhanced audio.
- Computes per-target WER and corpus WER.

Why this test exists:

- It evaluates all available target speakers, rather than only one selected
  dominant stream.
- It checks whether predicted DOAs are close enough for speaker-specific DSENet
  extraction.
- It is useful when you want a fuller offline WER report across multiple target
  speakers.

Important interpretation:

- Poor WER may come from wrong IPDNET DOAs, DSENet enhancement failure, or
  Whisper errors on enhanced audio.
- Because each target is matched to the nearest predicted DOA, this script is a
  good diagnostic for whether the predicted DOA set contains usable candidates.
- It does not model the final robot behavior of choosing only one dominant
  output.

Typical use:

```bash
python pipeline_full.py \
  --mic_dir ../../data/dataset_4mic_3spk/Eval/mic \
  --text_dir ../../data/dataset_4mic_3spk/Eval/text \
  --out_dir results/pipeline_full \
  --whisper_model small \
  --save_enhanced
```

Main outputs:

```text
pipeline_whisper_<model>_wer_details.csv
pipeline_whisper_<model>_wer_summary.json
```

## `pipeline_1asr.py`

This is the dominant-speaker one-ASR offline evaluation script.

What it does:

- Loads IPDNET, DSENet, and Whisper once.
- Processes one saved scene at a time.
- Predicts three DOAs with IPDNET.
- Enhances the predicted DOA candidates with DSENet.
- Selects the loudest enhanced stream by RMS.
- Runs Whisper only once on that selected stream.
- Scores the transcript against the dominant speaker spk1 text.
- Reports selected DOA error against the dominant spk1 ground-truth DOA.

Why this test exists:

- It tests the practical "one selected speaker -> one ASR call" path.
- It is closer to the intended audition behavior than `pipeline_full.py`.
- It measures whether loudest-enhanced selection tends to pick the dominant
  speaker.

Important interpretation:

- This script can produce high WER if the selected loudest enhanced stream is
  the wrong speaker, even when that stream is intelligible.
- The `selected_doa_error_deg` field is important for separating wrong-speaker
  selection from ASR quality.
- Use this when evaluating dominant-speaker extraction on full saved scenes.

Typical use:

```bash
python pipeline_1asr.py \
  --mic_dir ../../data/dataset_4mic_3spk/Eval/mic \
  --clean_dir ../../data/dataset_4mic_3spk/Eval/clean \
  --text_dir ../../data/dataset_4mic_3spk/Eval/text \
  --out_dir results/pipeline_1asr \
  --whisper_model small \
  --dse_batch_size 1 \
  --save_enhanced
```

Main outputs:

```text
pipeline_whisper_<model>_wer_details_1asr.csv
pipeline_whisper_<model>_wer_summary_1asr.json
```

## `pipeline_1asr_4s.py`

This is the chunked dominant-speaker one-ASR offline evaluation script.

What it does:

- Processes saved 4-second multichannel chunks one at a time.
- Runs IPDNET on each chunk.
- Enhances predicted DOA candidates with DSENet.
- Selects the loudest enhanced candidate per chunk.
- Runs Whisper once per selected chunk.
- Concatenates chunk hypotheses by scene `fileid`.
- Computes scene-level WER against dominant spk1 text.

Why this test exists:

- It mimics a chunked real-time pipeline while still running offline on saved
  data.
- It shows how per-4s decisions accumulate into full-scene ASR quality.
- It is useful for comparing saved-chunk behavior with online streaming
  experiments.

Important interpretation:

- Per-chunk selection can switch speakers across a scene, which may hurt
  scene-level WER even if individual chunks sound reasonable.
- The chunk details CSV is useful for finding which chunks had bad DOA
  predictions or wrong loudest-speaker choices.
- This is usually the best offline script for diagnosing the 4-second chunk
  design.

Typical use:

```bash
python pipeline_1asr_4s.py \
  --mic_dir ../../data/dataset_4mic_3spk_4s_full/Eval/mic_4s \
  --clean_dir ../../data/dataset_4mic_3spk_4s_full/Eval/clean \
  --text_dir ../../data/dataset_4mic_3spk_4s_full/Eval/text \
  --out_dir results/pipeline_1asr_4s \
  --whisper_model small \
  --dse_batch_size 1 \
  --save_enhanced
```

Main outputs:

```text
pipeline_whisper_<model>_chunk_details_1asr_4s.csv
pipeline_whisper_<model>_scene_wer_1asr_4s.csv
pipeline_whisper_<model>_wer_summary_1asr_4s.json
```

## `pipeline_realtime_full.py`

This is a timing-only offline realtime benchmark.

What it does:

- Loads IPDNET, DSENet, and Whisper once.
- Processes saved multichannel WAV scenes.
- Predicts DOAs with IPDNET.
- Runs DSENet on the predicted DOA batch.
- Sends one enhanced stream, controlled by `--whisper_index`, to Whisper.
- Records IPDNET, DSENet, Whisper, total time, and realtime factors.

Why this test exists:

- It measures whether the offline IPDNET -> DSENet -> Whisper chain is fast
  enough relative to audio duration.
- It does not require ground-truth text.
- It is useful when the question is latency rather than WER.

Important interpretation:

- This script is not a speaker-selection or WER benchmark.
- The chosen Whisper input is controlled by index, not by loudness or
  ground-truth matching.
- Use it to check throughput, batching, CUDA behavior, and rough realtime
  feasibility.

Typical use:

```bash
python pipeline_realtime_full.py \
  --mic_dir ../../data/dataset_4mic_3spk_4s/test/mic \
  --out_dir results/pipeline_realtime_full \
  --whisper_model small \
  --dse_batch_size 3 \
  --whisper_index 0 \
  --save_enhanced
```

Main outputs:

```text
pipeline_realtime_<model>_details.csv
pipeline_realtime_<model>_summary.json
```

## `pipeline_realtime_1asr.py`

This is the dominant-speaker timing benchmark with one ASR call.

What it does:

- Loads IPDNET, DSENet, and Whisper once.
- Processes one saved scene at a time.
- Predicts three DOAs with IPDNET.
- Enhances all predicted candidates.
- Selects the loudest enhanced output.
- Runs Whisper once on that selected output.
- Records selected DOA error, timing, realtime factor, and under-realtime rate.

Why this test exists:

- It combines the practical one-ASR selection logic with realtime timing
  metrics.
- It can be used before full WER evaluation to check whether the pipeline is
  computationally feasible.
- It reports whether the loudest selected DOA matches the dominant spk1 DOA.

Important interpretation:

- This script focuses on timing and selected DOA accuracy, not transcript WER.
- It is useful for validating the compute cost of the full frontend plus one
  ASR call.
- If `under_realtime_rate` is low, the pipeline is too slow for realtime use
  with the current model sizes or batch settings.

Typical use:

```bash
python pipeline_realtime_1asr.py \
  --mic_dir ../../data/dataset_4mic_3spk_4s/Eval/mic \
  --clean_dir ../../data/dataset_4mic_3spk_4s/Eval/clean \
  --out_dir results/pipeline_realtime_1asr \
  --whisper_model small \
  --save_enhanced
```

Main outputs:

```text
pipeline_realtime_<model>_details_1asr.csv
pipeline_realtime_<model>_summary_1asr.json
```

## `eval_doa_wer.py`

This is a small WER-vs-DOA-error analysis helper.

What it does:

- Loads a WER details CSV.
- Converts `selected_doa_error_deg` and `wer` to numeric values.
- Filters rows whose selected DOA error is below `--doa_threshold_deg`.
- Prints the number and percentage of selected rows.
- Prints mean sample WER for the filtered subset.
- If `edit_distance` and `ref_words` exist, also prints corpus WER.

Why this test exists:

- It answers: "How good is WER when the selected DOA is close enough to the
  dominant speaker?"
- It helps separate DOA/selection failure from downstream enhancement and ASR
  failure.

Important interpretation:

- If WER is still high for low-DOA-error rows, the bottleneck may be DSENet or
  Whisper rather than IPDNET selection.
- If WER improves strongly under the threshold, wrong DOA selection is a major
  source of failure.
- Some result CSVs may need `encoding="gb18030"` if they were saved or edited
  with a Windows Chinese code page.

Typical use:

```bash
python eval_doa_wer.py \
  --csv_path results/pipeline_1asr/pipeline_whisper_small_wer_details_1asr.csv \
  --doa_threshold_deg 40
```

## `plot_doa_error_dist.py`

This is a plotting helper for selected DOA error.

What it does:

- Loads a details CSV.
- Reads `selected_doa_error_deg`.
- Keeps valid errors between 0 and 180 degrees.
- Plots a histogram with 5-degree bins.
- Adds a KDE curve when enough data is available.
- Saves the figure to `--output_path`.

Why this test exists:

- It makes the selected-DOA error distribution easier to inspect visually.
- It helps reveal whether failures are concentrated near small errors, random
  over the circle, or dominated by wrong-speaker selections.

Important interpretation:

- A sharp peak near 0 degrees means loudest selection often chooses the correct
  dominant speaker.
- A wide or multi-modal distribution suggests unstable SSL predictions,
  ambiguous speaker selection, or a mismatch between selected enhanced output
  and dominant spk1 ground truth.

Typical use:

```bash
python plot_doa_error_dist.py \
  --csv_path results/pipeline_realtime_1asr/pipeline_realtime_small_details_1asr.csv \
  --output_path selected_doa_error_hist_kde_0_180.png
```

## Recommended Diagnostic Order

Use the scripts as a ladder:

```text
1. pipeline_realtime_full.py
   Check basic end-to-end compute time without needing transcripts.

2. pipeline_realtime_1asr.py
   Check one-ASR dominant-speaker timing and selected DOA accuracy.

3. pipeline_1asr.py
   Evaluate full-scene dominant-speaker WER.

4. pipeline_1asr_4s.py
   Evaluate 4-second chunked dominant-speaker WER and per-chunk selection.

5. pipeline_full.py
   Evaluate all target speakers using nearest predicted DOA matching.

6. eval_doa_wer.py and plot_doa_error_dist.py
   Analyze whether WER failures are mostly caused by bad selected DOA.
```
