python ablation_study/baseline/HARK_n_loc+sep.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/loc+sep.n \
  --hark_runner batchflow \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 50