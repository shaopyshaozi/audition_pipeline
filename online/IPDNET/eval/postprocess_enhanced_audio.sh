python postprocess_enhanced_audio.py   results/pipeline_streaming_4schunks_full_respeaker/concatenated_enhanced  \
    --noise_reduce_db 500 \
    --compressor_ratio 1 \
    --makeup_db 0 \
    --target_rms_dbfs -15 \
    --noise_percentile 60 \
    --highpass_hz 100  \
    --noise_profile_percentile 95


python postprocess_enhanced_audio.py \
  results/pipeline_streaming_4schunks_full_respeaker/concatenated_enhanced/enhanced_fileid_0_concatenated_loudest.wav \
  --chunk_sec 4 \
  --noise_reduce_db 500 \
  --noise_percentile 30 \
  --noise_profile_percentile 85 \
  --highpass_hz 100 \
  --compressor_ratio 1 \
  --makeup_db 10 \
  --target_rms_dbfs -15 \
  --peak_limit 0.95

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