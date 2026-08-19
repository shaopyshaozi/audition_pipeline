# Data Folder Overview

This folder contains the scripts used to create the synthetic 3-speaker, 4-microphone evaluation datasets used by the audition pipeline. The generated datasets follow a common `Eval` split layout with `clean`, `mic`, and `text` subfolders.

## Current Folder Map

| Path | Current contents | Created by |
| --- | --- | --- |
| `dataset_4mic_3spk/Eval` | 1200 clean wavs, 1200 mic wavs, 1200 transcript txt files | `data_creation_4mics.py` |
| `dataset_4mic_3spk_4s_full/Eval` | Same 1200 full-length clean/mic/text files as `dataset_4mic_3spk/Eval`, plus 1740 `clean_4s` wavs and 5220 `mic_4s` wavs | Full-length files are the same file set as `dataset_4mic_3spk`; 4-second subfolders are created by `4s_full_clean.py` and `4s_full_mic.py` |
| `dataset_doa_sep_3spk/Eval` | 1500 clean wavs, 1500 mic wavs, 1500 transcript txt files, `metadata.csv`, `generation_config.json` | `data_creation_DoA_separation.py` |

## Source Data

The dataset creation scripts draw speech and noise from external roots:

| Input | Default path in scripts | Purpose |
| --- | --- | --- |
| LibriSpeech | `.../DSENet/data/LibriSpeech` | Source speech clips and transcripts. Each generated scene samples three clips from distinct speakers with similar duration. |
| DEMAND | `.../DSENet/data/DEMAND` | Independent noise source. Noise is cropped or repeated to match the longest sampled speech clip. |

The scripts read audio with `soundfile`, convert multi-channel input to mono when needed, and resample to 16 kHz by default.

## Scripts

### `data_creation_4mics.py`

Creates the standard 3-speaker, 4-microphone ASR evaluation dataset.

Default command:

```bash
python data_creation_4mics.py
```

Important default arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--output_root` | `.../audition_pipeline/data/dataset_4mic_3spk` | Dataset folder to create. |
| `--eval_items` | `1200` | Number of target items written to `Eval`. |
| `--sample_rate` | `16000` | Output sample rate. |
| `--seed` | `40` | Random seed. |
| `--ref_mic` | `0` | Reference microphone used for clean target audio. |

How each scene is created:

1. Sample three LibriSpeech clips from distinct speakers whose durations differ by at most 5 seconds.
2. Scale speaker 1 to `-15 dB RMS`; scale speakers 2 and 3 randomly between `-25` and `-20 dB RMS`.
3. Sample one DEMAND noise file, tile it if it is too short, crop it to the longest speech duration, and scale it between `-35` and `-30 dB RMS`.
4. Sample a shoebox room:
   - width: 6 to 9 m
   - depth: 6 to 9 m
   - height: 3 m
   - RT60: 0.3 to 0.5 s
5. Place a ReSpeaker-style circular 4-mic array at the room center, height 1 m, radius 0.031 m. Microphone angles are 45, 135, 225, and 315 degrees.
6. Sample three source directions with at least 30 degrees pairwise DoA separation, then place each speaker approximately 2 m from the microphone center along its DoA.
7. Place the DEMAND noise source at a random valid room position.
8. Use `pyroomacoustics` to simulate:
   - one 4-channel mixture containing the three speakers plus noise
   - one mono reverberant clean reference per speaker at reference mic 0
9. Peak-normalize mixture and clean targets to avoid clipping.
10. Write one target item per speaker until `--eval_items` is reached.

Output layout:

```text
dataset_4mic_3spk/
  Eval/
    clean/
    mic/
    text/
```

File naming:

```text
clean/clean_fileid_<scene_id>_doa<target_doa>_spk<speaker_id>.wav
mic/mic_fileid_<scene_id>_doa<target_doa>_3spk.wav
text/text_fileid_<scene_id>_doa<target_doa>_spk<speaker_id>.txt
```

Notes:

- `scene_id` identifies the acoustic scene.
- `target_doa` is the integer DoA of the current target speaker.
- `speaker_id` is `1`, `2`, or `3`.
- The same underlying 4-channel mixture is written once per target speaker, under a DoA-specific mic filename, so downstream code can reconstruct the target from the filename.
- The transcript text is pulled from the matching LibriSpeech `.trans.txt` file.

### `data_creation_DoA_separation.py`

Creates a controlled DoA-separation stress-test dataset. It imports most room, audio, simulation, and file-writing utilities from `data_creation_4mics.py`, then changes the speaker placement and metadata logic.

Default command:

```bash
python data_creation_DoA_separation.py
```

Important default arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--output_root` | `.../audition_pipeline/data/dataset_doa_sep_3spk` | Dataset folder to create. |
| `--doa_separations` | `5,10,15,20,30` | Target-to-nearest-interferer DoA separations to test. |
| `--items_per_sep` | `100` | Number of scenes for each DoA separation. |
| `--split_name` | `Eval` | Output split name. |
| `--sample_rate` | `16000` | Output sample rate. |
| `--seed` | `42` | Random seed. |
| `--source_radius_m` | `2.0` | Speaker distance from microphone center where room boundaries allow it. |
| `--speaker_rms_db_min` | `-22.0` | Minimum speaker RMS level. |
| `--speaker_rms_db_max` | `-18.0` | Maximum speaker RMS level. |
| `--randomize_nearest_side` | off | If enabled, speaker 2 may be at `theta - separation`; otherwise it is at `theta + separation`. |

How each scene is created:

1. Iterate over each requested DoA separation.
2. For each separation, create `items_per_sep` scenes.
3. Sample three LibriSpeech clips from distinct speakers with similar duration.
4. Scale all three speakers to near-equal RMS, randomly between `-22` and `-18 dB RMS`.
5. Sample one DEMAND noise file, tile or crop it to the speech duration, and scale it between `-35` and `-30 dB RMS`.
6. Sample the same style of room and ReSpeaker 4-mic geometry as `data_creation_4mics.py`.
7. Sample target direction `theta`.
8. Place speakers in a controlled angular layout:
   - speaker 1: target at `theta`
   - speaker 2: nearest interferer at `theta + condition_doa_sep_deg`, or randomly plus/minus if `--randomize_nearest_side` is used
   - speaker 3: far interferer at `theta + 180`
9. Simulate the 4-channel mixture and mono reference-clean target for each speaker.
10. Write all three speakers as target items for each scene.
11. Write `metadata.csv` with one row per target item.
12. Write `generation_config.json` with the generation settings.

Output layout:

```text
dataset_doa_sep_3spk/
  Eval/
    clean/
    mic/
    text/
    metadata.csv
    generation_config.json
```

File naming:

```text
clean/clean_fileid_<scene_id>_doa<target_doa>_sep<doa_sep>_spk<speaker_id>.wav
mic/mic_fileid_<scene_id>_doa<target_doa>_sep<doa_sep>_3spk.wav
text/text_fileid_<scene_id>_doa<target_doa>_sep<doa_sep>_spk<speaker_id>.txt
```

Metadata highlights:

| Column group | Meaning |
| --- | --- |
| `scene_id`, `split`, `target_speaker_id`, `target_item_doa` | Item identity and target speaker. |
| `condition_doa_sep_deg` | Intended target-to-nearest-interferer DoA separation condition. |
| `target_to_spk2_sep_deg`, `target_to_spk3_sep_deg`, `min_pairwise_sep_deg` | Realized angular separations after integer DoA calculation. |
| `target_doa_requested`, `spk2_doa_requested`, `spk3_doa_requested` | Floating-point requested DoAs before rounding. |
| `spk1_doa`, `spk2_doa`, `spk3_doa` | Integer DoAs computed from simulated source positions. |
| `room_w`, `room_d`, `room_h`, `rt60`, `source_radius_m`, `mic_center_*` | Room and array geometry. |
| `spk*_source_file`, `spk*_rms_db` | Source LibriSpeech files and sampled speaker levels. |
| `clean_file`, `mic_file`, `text_file` | Written output paths. |

With the current defaults, the dataset has 5 separation conditions x 100 scenes x 3 target speakers = 1500 target items.

### `4s_full_clean.py`

Splits only speaker-1 clean reference files from `dataset_4mic_3spk_4s_full/Eval/clean` into fixed 4-second chunks.

Default command:

```bash
python 4s_full_clean.py
```

How it creates `clean_4s`:

1. Set `ROOT` to `data/dataset_4mic_3spk_4s_full/Eval`.
2. Read full-length wavs from `ROOT/clean`.
3. Keep only files whose name contains `spk1` or `speaker1`.
4. For each selected clean wav:
   - compute `clip_len = 4.0 * sample_rate`
   - split the file into consecutive 4-second chunks
   - use `ceil(length / clip_len)`, so the final partial chunk is kept
   - pad the final chunk with zeros if it is shorter than 4 seconds
5. Write chunks to `ROOT/clean_4s`.

Output naming:

```text
clean_4s/<original_clean_stem>_<chunk_index>.wav
```

Example:

```text
clean_fileid_0_doa73_spk1_1.wav
```

Notes:

- Only `spk1` clean files are chunked.
- This is why the current folder has 1740 clean chunks, not 3 times that number.
- The original `clean`, `mic`, and `text` folders must already exist before this script runs.

### `4s_full_mic.py`

Intended to split matching full-length microphone mixtures from `dataset_4mic_3spk_4s_full/Eval/mic` into fixed 4-second chunks, using the corresponding speaker-1 clean duration as the time limit.

Default command:

```bash
python 4s_full_mic.py
```

How it creates `mic_4s`:

1. Set `ROOT` to `data/dataset_4mic_3spk_4s_full/Eval`.
2. Build a map from `fileid` to the matching `spk1` clean file in `ROOT/clean`.
3. Group all full-length mic wavs in `ROOT/mic` by `fileid`.
4. For each `fileid`:
   - get the `spk1` clean length and sample rate
   - compute the number of 4-second chunks from the `spk1` clean length
   - read every mic wav for that same `fileid`
   - require the mic sample rate to match the clean sample rate
   - crop each mic signal to the `spk1` clean length
   - split into 4-second chunks
   - pad the final chunk with zeros if needed
5. Write chunks to `ROOT/mic_4s`.

Output naming:

```text
mic_4s/<original_mic_stem>_<chunk_index>.wav
```

Example:

```text
mic_fileid_0_doa169_3spk_1.wav
```

Notes:

- All mic files for a scene are chunked, but their chunk count is determined by the speaker-1 clean length for that `fileid`.
- The current folder has 5220 mic chunks, exactly 3 x 1740, because each scene has three target-specific mic filenames.
- The script currently has a typo in this line:

```python
OUTPUT_MIC_DIR = ROOT / "mic_4s
```

It should be:

```python
OUTPUT_MIC_DIR = ROOT / "mic_4s"
```

The existing `mic_4s` folder was presumably produced after using the intended corrected line.

## Dataset Relationships

### `dataset_4mic_3spk`

This is the base 3-speaker dataset. It is created directly by `data_creation_4mics.py` with its default `--output_root`.

### `dataset_4mic_3spk_4s_full`

This folder is the 4-second chunked version of the base dataset.

The current full-length `clean`, `mic`, and `text` file names match `dataset_4mic_3spk/Eval` exactly. The 4-second scripts do not create those full-length folders themselves; they require them as input. In practice, create this folder by either:

1. running `data_creation_4mics.py` with `--output_root` set to `data/dataset_4mic_3spk_4s_full`, or
2. copying the `Eval/clean`, `Eval/mic`, and `Eval/text` folders from `dataset_4mic_3spk`.

Then run:

```bash
python 4s_full_clean.py
python 4s_full_mic.py
```

after fixing the quote typo in `4s_full_mic.py`.

### `dataset_doa_sep_3spk`

This is a separate controlled experiment dataset. It is created directly by `data_creation_DoA_separation.py` and is not derived from `dataset_4mic_3spk`.

The key difference is the speaker geometry:

- `dataset_4mic_3spk` samples three DoAs with a minimum pairwise separation.
- `dataset_doa_sep_3spk` fixes the target-to-nearest-interferer separation to explicit test conditions and stores the realized geometry in metadata.

## Common File Semantics

| Subfolder/file | Meaning |
| --- | --- |
| `clean/*.wav` | Mono reverberant target-speaker reference at the configured reference mic. |
| `mic/*.wav` | Four-channel ReSpeaker-style simulated mixture containing three speakers plus DEMAND noise. |
| `text/*.txt` | Ground-truth transcript for the target speaker. |
| `clean_4s/*.wav` | Four-second chunks of speaker-1 clean references, padded on the final chunk if needed. |
| `mic_4s/*.wav` | Four-second chunks of mic mixtures, cropped to speaker-1 length and padded on the final chunk if needed. |
| `metadata.csv` | Per-target-item metadata for the controlled DoA-separation dataset. |
| `generation_config.json` | Generation settings for the controlled DoA-separation dataset. |

## Dependencies

The generation scripts use:

- `numpy`
- `soundfile`
- `scipy`
- `pyroomacoustics`
- `torchaudio`
- `tqdm`

These are consistent with the top-level `requirements.txt` and the imports inside the scripts.
