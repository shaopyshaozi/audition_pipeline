python3 ablation_study/non_dominant/respeaker_streaming_full_spk.py \
  --mode both \
  --hark_network ablation_study/baseline/HARK/loc+sep.n \
  --tf_zip ablation_study/baseline/HARK/respeaker4_tf_5deg.zip \
  --hark_runner batchflow \
  --whisper_model small


python3 ablation_study/non_dominant/respeaker_streaming_4schunks_minor_spk.py \
  --mode eval_official \
  --whisper_model small


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