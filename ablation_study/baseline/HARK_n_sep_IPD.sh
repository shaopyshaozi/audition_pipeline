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