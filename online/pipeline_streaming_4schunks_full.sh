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


python online/pipeline_streaming_4schunks_full.py \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --max_items 50 \
  --stream_realtime