# Evaluation Python Files Overview

This folder contains the Respeaker evaluation scripts used to test the online
audition pipeline:

```text
Respeaker 4-mic audio -> 4s chunks -> SSL/IPDNET -> DSENet -> ASR/Whisper
```

The main goal of these tests is to understand whether the full system can work
for a real-time 3-speaker Respeaker setting, and to isolate which module is
responsible when WER or enhanced audio quality becomes poor.

The most important recent baseline is the raw/no-enhancement run:

```text
No SSL + no DSENet enhancement -> no-insertion WER: 42.83%
```

This should serve as a practical goal for the enhancement pipeline. If the
IPDNET -> DSENet pipeline performs worse than this raw baseline, then the
enhancement stage is not doing the right thing for the current real Respeaker
condition.

## Common Assumptions

Most pipeline scripts expect Respeaker recordings named like:

```text
fileid_<id>_sources_3_<doa1>_<doa2>_<doa3>.wav
```

The long recording is split into 4s chunks in memory. The chopped raw chunks are
not normally saved unless the script explicitly saves enhanced outputs.

Common outputs under `--out_dir`:

```text
pipeline_realtime_enhanced/
  selected enhanced 4s chunks

pipeline_realtime_enhanced_all/
  all DSENet enhanced candidates, if --save_all_enhanced is used

concatenated_enhanced/
  selected enhanced chunks concatenated back to one WAV per fileid

pipeline_streaming_<model>_details_*.csv
  per-4s-chunk timing, DoA, selection, and transcript assignment

pipeline_streaming_<model>_scene_wer_*.csv
  per-fileid WER summary

pipeline_streaming_<model>_summary_*.csv/json
  corpus-level WER, DoA accuracy, realtime timing, and aggregate metrics

pipeline_streaming_<model>_transcripts_4s_full.jsonl
  raw streaming ASR transcript segments
```

When `--out_dir` is relative, it is relative to the directory where you run the
command. For example, running from `online/IPDNET/eval` with:

```bash
--out_dir results/my_run
```

saves into:

```text
online/IPDNET/eval/results/my_run
```

## `pipeline_streaming_4schunks_full.py`

This is the main non-post-processed dominant-speaker streaming pipeline.

What it does:

- Loads IPDNET and DSENet once.
- Splits each long Respeaker recording into 4s chunks.
- Runs IPDNET on each chunk to estimate source DoAs.
- Runs DSENet once per chunk with a batch of predicted DoAs.
- Selects the dominant speaker by choosing the loudest enhanced output.
- Sends only that selected enhanced signal to SimulStreaming Whisper.
- Evaluates against dominant speaker text references, usually `0.txt` to `99.txt`.

Why this test exists:

- It is the closest baseline to the intended robot pipeline.
- It measures the full online chain: SSL, enhancement, dominant selection, ASR,
  WER, and realtime latency.
- It answers: "Does the complete pipeline work for the dominant speaker without
  extra audio cleanup?"

Important interpretation:

- Bad WER can come from any stage: wrong SSL DoA, wrong dominant selection,
  poor DSENet enhancement, ASR streaming segmentation, or real recording domain
  mismatch.
- Use the DoA correctness metrics and WER-when-DoA-correct fields to separate
  SSL failure from enhancement/ASR failure.

What we found:

- The pipeline is fast enough for the online setting; the 4s chunk design is
  realistic for robot response time.
- Dominant selection by enhanced-output loudness can work reasonably, but it is
  fragile when DSENet produces a clearer output for the wrong speaker.
- WER can become misleading when the selected enhanced audio belongs to another
  speaker. The ASR text may be correct for that wrong speaker, but it becomes
  insertions/substitutions against the dominant-speaker reference.
- The no-insertion WER and WER-when-DoA-correct metrics were added to make this
  evaluation more objective.
- Real Respeaker audio is much harder than the synthetic pipeline setting, even
  when the system is fast enough.

- **Very poor performance, 75% of word error rate with no insertion**

Typical use:

```bash
python pipeline_streaming_4schunks_full.py \
  --respeaker_dir Respeaker_recordings \
  --out_dir results/pipeline_streaming_4schunks_full_respeaker \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --stream_realtime \
  --respeaker_source_count 3 \
  --save_enhanced \
  --max_items 20 \
  --input_gain 3.0
```

## `pipeline_streaming_4schunks_full_post_processed.py`

This is the dominant-speaker pipeline with built-in post-processing.

What it does:

- Runs the same SSL -> DSENet -> loudest-speaker selection structure as
  `pipeline_streaming_4schunks_full.py`.
- Applies fast DSP post-processing to the selected enhanced chunk before ASR,
  saving, and concatenation.
- Can also save all enhanced candidates with `--save_all_enhanced`.
- When saving all candidates, filenames are ordered/aligned by closeness to the
  ground-truth DoAs in the input filename.

Default post-processing behavior:

```text
high-pass around 100 Hz
spectral noise suppression
compressor disabled by default, ratio = 1
preserve each chunk's original enhanced RMS
peak limit at 0.95
```

Why this test exists:

- It tests whether cheap, realtime-friendly post-processing can improve ASR
  performance or listening quality without retraining DSENet.
- It answers: "Can we make the enhanced audio clearer after DSENet using only
  lightweight audio processing?"

Important interpretation:

- If this improves listening quality but worsens WER, the post-processing may be
  changing speech cues that Whisper needs.
- If it adds artifacts, the issue is probably not post-processing alone; DSENet
  may already be producing unstable outputs.
- Disable post-processing with:

```bash
--no-postprocess_enhanced
```

What we found:

- The standalone-tuned post-processing settings sounded best when applied per 4s
  chunk and when each chunk's original enhanced RMS was preserved.
- Aggressive denoising could make audio less quiet, but it sometimes introduced
  splashy/noisy artifacts.
- **Increasing loudness or adding post-processing did not reliably improve WER.**
  **In some cases it made WER worse because artifacts or wrong-speaker speech became easier for ASR to transcribe.**
- This suggests **post-processing is not the main solution**. It can polish a usable
  enhanced signal, but it cannot fix a DSENet output that is already dominated
  by noise, leakage, or wrong-speaker extraction.
- `--save_all_enhanced` is useful for listening to every DSENet candidate and
  checking whether the selected output is actually the intended speaker.

## `pipeline_streaming_4schunks_full_post_processed_minor_spk.py`

This is the source-2/minor-speaker pipeline using predicted SSL DoAs.

What it does:

- Runs IPDNET to predict DoAs for each 4s chunk.
- Runs DSENet with the predicted DoAs.
- Instead of selecting the loudest enhanced output, it selects the candidate
  whose predicted DoA is closest to filename DoA2.
- Transcribes that selected source-2 enhanced audio.
- Evaluates against source-2 transcripts from:

```text
Respeaker_real/text/src2
```

with filename offset:

```text
200.txt -> fileid_0
201.txt -> fileid_1
...
299.txt -> fileid_99
```

Why this test exists:

- Dominant-speaker evaluation can hide problems with quieter/background speakers.
- This test checks whether the pipeline can extract a non-dominant target when
  the target source is known to be DoA2.
- It answers: "Can the SSL + DSENet pipeline transcribe source 2, not just the
  loudest speaker?"

Important interpretation:

- This script intentionally uses filename DoA2 as the target for selecting among
  SSL-predicted DSENet candidates.
- It is not the blind dominant-speaker robot behavior. It is a controlled
  source-2 diagnostic.
- If this fails but the dominant pipeline works, the system may be biased toward
  stronger speakers or DSENet may struggle with low-SNR targets.

What we found:

- Minor/source-2 evaluation is much harder than dominant-speaker evaluation.
- Using loudness is not suitable for this test, because the target is explicitly
  source 2 rather than the loudest speaker.
- Selecting the enhanced candidate closest to filename DoA2 makes this a useful
  diagnostic for non-dominant speaker extraction.
- **This script still performs poorly, there are still two possible causes:** 
  **SSL may predict poor DoAs, or DSENet may fail even when a nearby DoA candidate exists.**
- This motivated the GT-DoA version below.

Typical use:

```bash
python pipeline_streaming_4schunks_full_post_processed_minor_spk.py \
  --respeaker_dir Respeaker_recordings \
  --out_dir results/pipeline_streaming_4schunks_full_respeaker_minor_spk \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --stream_realtime \
  --respeaker_source_count 3 \
  --save_enhanced \
  --max_items 20 \
  --input_gain 3.0
```

## `pipeline_streaming_4schunks_full_post_processed_minor_spk_gt.py`

This is the source-2/minor-speaker ground-truth-DoA pipeline.

What it does:

- Blocks SSL/IPDNET completely.
- Reads DoA2 directly from the input filename.
- Runs DSENet once per 4s chunk with only that ground-truth DoA2.
- Sends the resulting enhanced source-2 audio to streaming Whisper.
- Evaluates against `Respeaker_real/text/src2` using the same `200.txt` to
  `fileid_0` mapping.

Why this test exists:

- It isolates DSENet from SSL.
- If the normal minor-speaker pipeline fails, this script asks whether the
  failure is due to wrong SSL DoAs or due to DSENet/enhancement itself.
- It answers: "If DoA is perfect, can DSENet extract source 2 from real
  Respeaker recordings?"

Important interpretation:

- If this GT-DoA pipeline still produces mostly noise, SSL is not the main
  bottleneck.
- That strongly suggests DSENet is mismatched to the real Respeaker condition or
  underperforming for the target setting.
- This is the most useful diagnostic before deciding whether to fine-tune or
  retrain DSENet.

What we found:

- **Even with SSL removed and filename DoA2 used directly, the real Respeaker enhanced audio can still be very poor or almost pure noise.**
- **This strongly suggests the bottleneck is not only SSL. The current DSENet checkpoint itself is likely mismatched to the real 3-speaker Respeaker condition.**
- The likely mismatch is training data: the current checkpoint came from a
  synthetic multi-speaker setting that does not fully represent real Respeaker
  microphone response, real room acoustics, quiet input gain, and partial
  speaker activity inside 4s chunks.
- This test supports the next research step: fine-tune or retrain DSENet on a
  more realistic 4-mic 3-speaker simulation dataset, ideally with variable
  onset/offset, source loudness variation, wider room/noise conditions, and
  Respeaker-like gain/noise.

Typical use:

```bash
python pipeline_streaming_4schunks_full_post_processed_minor_spk_gt.py \
  --respeaker_dir Respeaker_recordings \
  --out_dir results/pipeline_streaming_4schunks_full_respeaker_minor_spk_gt \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --stream_realtime \
  --respeaker_source_count 3 \
  --save_enhanced \
  --max_items 20 \
  --input_gain 3.0
```

## `pipeline_streaming_4schunks_full_noenhanced.py`

This is the raw Respeaker streaming ASR baseline.

What it does:

- Splits each long Respeaker recording into 4s chunks.
- Applies `--input_gain` to the raw multichannel chunk.
- By default, averages the Respeaker channels to mono.
- Sends that raw mono signal directly to SimulStreaming Whisper.
- Does not use SSL/IPDNET.
- Does not use DSENet enhancement.
- Keeps an optional `--frontend_mode full` path that can still run the old
  IPDNET -> DSENet -> loudest-enhanced frontend, but the important default mode
  is `--frontend_mode raw`.

Why this test exists:

- It checks whether the enhancement pipeline is actually helping.
- It gives a no-SSL/no-enhancement baseline using the same 4s chunking,
  streaming ASR, input gain, transcript assignment, and WER calculation as the
  full pipeline.
- It answers: "If we do nothing except stream the raw Respeaker signal, how good
  is ASR?"

Important interpretation:

- This file is not trying to separate speakers.
- It is a sanity-check baseline for the full system.
- If raw Respeaker audio gives lower WER than enhanced audio, the enhancement
  pipeline is damaging the ASR input or selecting/extracting the wrong content.
- This is especially important because a speech enhancement model can make audio
  sound more processed while still making ASR worse.

What we found:

- The no-SSL/no-enhancement raw baseline gives:

```text
no-insertion WER: 42.83%
```

- This should be treated as the minimum practical goal for the enhancement
  pipeline.
- The current IPDNET -> DSENet enhancement pipeline is not doing the right thing
  if it cannot beat this raw baseline.
- This finding strengthens the conclusion from the GT-DoA source-2 test: **the**
  **current DSENet checkpoint is likely mismatched to the real Respeaker setting,**
  **and the pipeline should be improved through a more realistic 3-speaker**
  **simulation dataset plus fine-tuning or retraining.**

Typical use:

```bash
python pipeline_streaming_4schunks_full_noenhanced.py \
  --respeaker_dir Respeaker_recordings \
  --out_dir results/pipeline_streaming_4schunks_full_noenhanced \
  --frontend_mode raw \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --stream_realtime \
  --respeaker_source_count 3 \
  --save_enhanced \
  --max_items 20 \
  --input_gain 3.0
```

## `postprocess_enhanced_audio.py`

This is a standalone post-processing tester for already saved enhanced WAVs.

What it does:

- Accepts either one WAV file or a folder of WAV files.
- Splits full audio into 4s chunks by default.
- Processes each chunk independently.
- Concatenates processed chunks back into a full WAV.
- Can also save individual processed chunks with `--save_chunks`.

Default chain:

```text
high-pass filter
spectral denoise
optional compressor
peak limiting
preserve original chunk RMS by default
optional global RMS normalization
```

Why this test exists:

- It lets you tune post-processing values without rerunning IPDNET, DSENet, or
  Whisper.
- It answers: "Can simple DSP improve already enhanced audio, and which settings
  sound best?"

Useful command:

```bash
python postprocess_enhanced_audio.py \
  results/pipeline_streaming_4schunks_full_respeaker/concatenated_enhanced \
  --chunk_sec 4 \
  --noise_reduce_db 500 \
  --noise_percentile 60 \
  --noise_profile_percentile 95 \
  --highpass_hz 100 \
  --compressor_ratio 1 \
  --makeup_db 0 \
  --preserve_chunk_rms \
  --preserve_chunk_rms_blend 1.0 \
  --peak_limit 0.95
```

Important interpretation:

- This script is for listening and tuning. It does not prove DSENet is good or
  bad by itself.
- If post-processing helps only slightly or creates artifacts, the source issue
  is likely upstream in DSENet enhancement quality.

What we found:

- The best-sounding setting so far was aggressive denoising with RMS
  preservation:

```bash
--chunk_sec 4 \
--noise_reduce_db 500 \
--noise_percentile 60 \
--noise_profile_percentile 95 \
--highpass_hz 100 \
--compressor_ratio 1 \
--makeup_db 0 \
--preserve_chunk_rms \
--preserve_chunk_rms_blend 1.0 \
--peak_limit 0.95
```

- Processing full audio directly gave different results from pipeline
  post-processing, because the pipeline operates on independent 4s chunks.
- Preserving chunk RMS helped avoid unnatural chunk-to-chunk loudness jumps.
- Very high `noise_reduce_db` values act like an extremely low spectral floor;
  once it is already very large, increasing it further may not make a practical
  difference.
- The tool is useful for tuning listening quality, but it did not solve the
  deeper DSENet generalization problem.

## Why These Tests Were Constructed

The scripts form a diagnostic ladder:

```text
1. Raw Respeaker baseline, no SSL and no enhancement
   Establishes the ASR baseline that enhancement must beat.

2. Dominant pipeline, no post-process
   Tests the intended online system in its cleanest form.

3. Dominant pipeline, with post-process
   Tests whether cheap realtime DSP can improve the selected enhanced audio.

4. Minor speaker with SSL
   Tests whether the system can handle a non-dominant target speaker.

5. Minor speaker with GT DoA
   Removes SSL as a variable and directly tests DSENet with perfect DoA.

6. Standalone post-processing
   Tunes DSP without rerunning the whole pipeline.
```

This separation matters because a high WER can have different causes:

```text
Bad SSL DoA
Wrong enhanced candidate selected
DSENet cannot extract the target
Post-processing damages speech cues
Streaming ASR segmentation issues
Real Respeaker domain mismatch
```

The GT-DoA source-2 script is especially important. If it fails on real
Respeaker recordings even with the correct filename DoA2, then the current
checkpoint is likely not robust enough for the final real 3-speaker Respeaker
setting. That supports fine-tuning or retraining DSENet on a more realistic
3-speaker simulation dataset.

The raw/no-enhancement baseline is equally important. It shows what happens when
the pipeline does not attempt separation at all. The current raw baseline result
is:

```text
no-insertion WER: 42.83%
```

Therefore, the enhancement pipeline should aim to beat `42.83%` no-insertion
WER. If it performs worse, the enhancement pipeline is not doing the right thing
for the current real Respeaker recordings.

## Recommended Comparison Matrix

For each run, compare:

```text
corpus_wer
corpus_no_insertion_wer
corpus_wer_cleaned
target_doa_accuracy
wrong_speaker_selection_rate
mean_selected_doa_error_deg
mean_frontend_compute_rtf
final_pipeline_lag_sec
saved enhanced audio listening quality
```

Suggested run folders:

```text
results/raw_noenhanced
results/dominant_no_post
results/dominant_post
results/minor_spk_ssl
results/minor_spk_gt
```

Keeping output folders separate avoids mixing old dominant-speaker files with
new minor-speaker or GT-DoA results.
