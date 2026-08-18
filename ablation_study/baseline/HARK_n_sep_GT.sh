python ablation_study/baseline/HARK_n_sep_GT.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/sep.n \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --hark_runner batchflow \
  --whisper_model small \
  --whisper_device cuda \
  --max_items 50