# Baseline Summary

This folder contains offline baselines for the 4-channel ReSpeaker, 3-speaker evaluation set. Most scripts evaluate Whisper WER, and scripts with enhancement/separation references also report SDR/SDRi and SI-SDR/SI-SDRi where clean references are available.

## Baseline Table

Official HARK scripts call HARK through `batchflow` and an HARK Designer `.n` network. They should be run inside the Ubuntu/WSL environment where HARK 3.6.0 is installed.

| Baseline | Script | Input | DOA / Selection | Enhancement / Separation | ASR | Final Evaluated Audio / Notes |
|---|---|---|---|---|---|---|
| No enhancement | `No_enhanced.py` | 1 fixed ReSpeaker channel | None | None | Whisper | Direct noisy-mixture ASR baseline, spk1 by default. |
| MVDR GT DOA | `MVDR.py` | 4-channel wav | GT DOA | MVDR beamforming | Whisper | Dominant-speaker/spk1 style baseline. |
| MVDR IPDNet DOA | `MVDR.py` | 4-channel wav | IPDNet predicted DOA closest to spk1 GT DOA | MVDR beamforming | Whisper | Classical spatial baseline with learned SSL, evaluated on spk1. |
| SepFormer | `SepFormer.py`, `SepFormer_libri.py`, `SepFormer_libri_gt.py` | 1 fixed channel | Oracle SI-SDR source matching | SpeechBrain SepFormer, 3-source separation | Whisper | Single-channel neural separation; source order is arbitrary. |
| Conv-TasNet | `Conv-TasNet_libri.py` | 1 fixed channel | Oracle SI-SDR source matching | Asteroid Conv-TasNet, 3-source separation | Whisper | Single-channel neural separation; source order is arbitrary. |
| HARK SSL + separation | `HARK_n_loc+sep.py` | 4-channel wav + `HARK/loc+sep.n` | HARK `LocalizeMUSIC` + tracker | Official HARK GHDSS | Whisper | WER-best HARK output in that scene. |
| HARK SSL + separation + post-filter | `HARK_n_loc+sep+pf.py` | 4-channel wav + `HARK/loc+sep+pf.n` | HARK `LocalizeMUSIC` + tracker | Official HARK GHDSS + post-filter network | Whisper | WER-best HARK output in that scene. |
| HARK GT-DOA separation | `HARK_n_sep_GT.py` | 4-channel wav + `HARK/sep.n` | GT DOAs injected into `ConstantLocalization` | Official HARK GHDSS | Whisper | spk-ordered output; final comparison is spk1. |
| HARK GT-DOA separation + post-filter | `HARK_n_sep+pf_GT.py` | 4-channel wav + `HARK/sep+pf.n` | GT DOAs injected into `ConstantLocalization` | Official HARK GHDSS + post-filter network | Whisper | spk-ordered output; final comparison is spk1. |
| HARK IPDNet-DOA separation | `HARK_n_sep_IPD.py` | 4-channel wav + `HARK/sep.n` | IPDNet predicted DOAs injected into `ConstantLocalization` | Official HARK GHDSS | Whisper | Output whose predicted DOA is closest to spk1 GT DOA. |
| HARK IPDNet-DOA separation + post-filter | `HARK_n_sep+pf_IPD.py` | 4-channel wav + `HARK/sep+pf.n` | IPDNet predicted DOAs injected into `ConstantLocalization` | Official HARK GHDSS + post-filter network | Whisper | Output whose predicted DOA is closest to spk1 GT DOA. |

HARK candidate outputs are stored under each `fileid_*` folder. For IPDNet-DOA HARK separation, files are labeled by predicted DOA, for example:

```text
enhanced_fileid_0_preddoa72_src1.wav
enhanced_fileid_0_preddoa168_src2.wav
enhanced_fileid_0_preddoa205_src3.wav
```

For GT-DOA HARK separation, files are labeled by the GT DOA and speaker id, for example:

```text
enhanced_fileid_0_doa73_spk1.wav
enhanced_fileid_0_doa169_spk2.wav
enhanced_fileid_0_doa206_spk3.wav
```

## Common Commands

No enhancement:

```bash
python ablation_study/baseline/No_enhanced.py --target_speaker_id 1
```

MVDR:

```bash
python ablation_study/baseline/MVDR.py --doa_source gt --mvdr_diag_load 1e-3
python ablation_study/baseline/MVDR.py --doa_source ipdnet --mvdr_diag_load 1e-3
```

Single-channel neural separation:

```bash
python ablation_study/baseline/SepFormer.py --target_speaker_id 1 --save_enhanced --max_items 50
python ablation_study/baseline/Conv-TasNet_libri.py --target_speaker_id 1 --save_enhanced --max_items 50
```

Official HARK, GT-DOA separation:

```bash
python ablation_study/baseline/HARK_n_sep_GT.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/sep.n \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --hark_runner batchflow \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 50
```

Official HARK, IPDNet-DOA separation:

```bash
python ablation_study/baseline/HARK_n_sep_IPD.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/sep.n \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --hark_runner batchflow \
  --ipd_ckpt Models/SSL/IPDNET/last-v1.ckpt \
  --ipd_device cuda \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 50
```

Official HARK, HARK SSL + separation:

```bash
python ablation_study/baseline/HARK_n_loc+sep.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/loc+sep.n \
  --hark_runner batchflow \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 50
```

## Interpretation Notes

- `--mode run_official` only runs HARK and writes separated wav files.
- `--mode eval_official` only evaluates existing HARK outputs.
- `--mode both` runs HARK first and then evaluates the generated outputs.
- HARK scripts require official HARK `batchflow`, a valid `.n` network, and `HARK/respeaker4_tf_5deg.zip`.
- HARK GT-DOA baselines are oracle spatial baselines because ground-truth DOAs are injected into `ConstantLocalization`.
- HARK IPDNet-DOA baselines are non-GT spatial baselines: IPDNet predicts DOAs, then the separated stream closest to spk1 GT DOA is used for the final spk1 comparison.
- SepFormer and Conv-TasNet use oracle SI-SDR matching because single-channel separation source order is arbitrary.
