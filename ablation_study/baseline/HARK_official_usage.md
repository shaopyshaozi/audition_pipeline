# Official PyHARK Baseline Usage

`HARK.py` now uses **official PyHARK** rather than a HARK Designer `.n` file
or `batchflow`.

The processing path is:

```text
4-channel wav
  -> PyHARK AudioStreamFromMemory
  -> PyHARK MultiFFT
  -> PyHARK LocalizeMUSIC
  -> PyHARK SourceTracker
  -> PyHARK GHDSS
  -> PyHARK Synthesize + SaveWavePCM
  -> Whisper small evaluation
```

## Check PyHARK

Run this in the same environment used for the baseline:

```bash
./venv_wsl/bin/python -c "import hark; import hark.node; print(hark)"
```

If this fails, install PyHARK/HARK 4 first. The HARK site describes PyHARK as
the official Python bindings that are imported with `import hark`.

## Required Transfer Functions

You still need HARK transfer function files for your ReSpeaker 4-mic geometry:

```text
localization TF: used by LocalizeMUSIC A_MATRIX
separation TF:  used by GHDSS TF_CONJ_FILENAME
```

They may be the same file if generated that way:

```bash
--localization_tf /path/to/respeaker4_tf.zip
```

or separate:

```bash
--localization_tf /path/to/respeaker4_localization_tf.zip \
--separation_tf /path/to/respeaker4_separation_tf.zip
```

## Run PyHARK + Evaluation

```bash
./venv_wsl/bin/python ablation_study/baseline/HARK.py \
  --mode both \
  --localization_tf /path/to/respeaker4_tf.zip \
  --whisper_model small \
  --whisper_device cuda \
  --source_selection oracle_sisdr
```

For a quick one-scene smoke test:

```bash
./venv_wsl/bin/python ablation_study/baseline/HARK.py \
  --mode both \
  --localization_tf /path/to/respeaker4_tf.zip \
  --whisper_model small \
  --whisper_device cuda \
  --source_selection oracle_sisdr \
  --max_items 1
```

## Run PyHARK Only

```bash
./venv_wsl/bin/python ablation_study/baseline/HARK.py \
  --mode run_pyhark \
  --localization_tf /path/to/respeaker4_tf.zip
```

Outputs are written to:

```text
ablation_study/baseline/results/HARK/official_hark_outputs/fileid_<N>/
```

The PyHARK runner uses `SaveWavePCM` with this basename:

```text
sep_{srcid}_
```

so the exact file names depend on HARK source IDs.

## Evaluate Existing PyHARK Outputs

```bash
./venv_wsl/bin/python ablation_study/baseline/HARK.py \
  --mode eval_official \
  --official_output_dir ablation_study/baseline/results/HARK/official_hark_outputs \
  --whisper_model small \
  --whisper_device cuda \
  --source_selection oracle_sisdr
```

`oracle_sisdr` resolves HARK source-order ambiguity by choosing the separated
stream with the best SI-SDR against the target clean source. If you export
per-source DOAs from PyHARK later, prefer:

```bash
--source_selection doa_csv --doa_csv /path/to/hark_output_doas.csv
```

## Main Files

```text
HARK.py                         batch run + Whisper/WER/SI-SDRi evaluation
pyhark_localizemusic_ghdss.py   one-file PyHARK LocalizeMUSIC + GHDSS runner
```

## Paper Label

Use:

```text
Official PyHARK: LocalizeMUSIC + SourceTracker + GHDSS + Whisper-small
```

If `--source_selection oracle_sisdr` is used, report it as oracle output
assignment.
