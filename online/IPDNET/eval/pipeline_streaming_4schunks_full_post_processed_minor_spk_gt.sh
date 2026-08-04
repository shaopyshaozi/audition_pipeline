python simulstreaming_whisper_server.py \
  --host localhost \
  --port 43001 \
  --language en \
  --task transcribe \
  --model_path ./small.pt \
  --min-chunk-size 2.0 \
  --audio_max_len 5 \
  --max_context_tokens 0 \
  --log-level WARNING 2>&1 | python clean_transcript.py


python pipeline_streaming_4schunks_full_post_processed_minor_spk_gt.py \
  --respeaker_dir Respeaker_real/mic \
  --out_dir results/pipeline_streaming_4schunks_full_respeaker_minor_spk_gt \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --stream_realtime \
  --respeaker_source_count 3 \
  --save_enhanced \
  --max_items 20 \
  --input_gain 1.0 \
  --no-postprocess_enhanced