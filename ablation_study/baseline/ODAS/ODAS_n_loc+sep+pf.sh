cd /home/shaozi/ucl/code/audition_pipeline

python3 ablation_study/baseline/ODAS/ODAS_n_loc+sep+pf.py \
  --mode both \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 1

python3 ablation_study/baseline/ODAS/ODAS_n_loc+sep+pf.py \
  --mode both \
  --whisper_model small \
  --whisper_device cuda \
  --skip_existing

python3 ablation_study/baseline/ODAS/ODAS_n_loc+sep+pf.py \
  --mode both \
  --whisper_model small \
  --whisper_device cuda \
  --skip_existing \
  --audio_stage separated
