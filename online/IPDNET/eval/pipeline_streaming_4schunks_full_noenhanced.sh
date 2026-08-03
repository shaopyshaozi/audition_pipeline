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

python pipeline_streaming_4schunks_full_noenhanced.py \
  --respeaker_dir Respeaker_recordings \
  --out_dir results/raw_respeaker \
  --input_gain 1.0 \
  --frontend_mode raw \
  --max_item 20 
