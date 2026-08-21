## Pyaudio connections, 2 terminals needed, one for each command below------------------------------------------------------------------------------

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

python pipeline_streaming_4schunks_full_respeaker_linux.py \
  --audio_source pyaudio \
  --respeaker_index 4 \
  --respeaker_rate 16000 \
  --sample_rate 16000 \
  --respeaker_channels 6 \
  --respeaker_mic_channels 1,2,3,4 \
  --chunk_seconds 4 \
  --skip_asr_policy silence \
  --raw_input_gain 1 \
  --save_raw_chunks \
  --save_enhanced

python pipeline_streaming_4schunks_full_respeaker_linux.py --list_audio_devices

#-----------------------------------------------------------------------------------------------------------------------

python pipeline_streaming_4schunks_full_respeaker_linux.py \
  --streaming_mode external \
  --streaming_host localhost \
  --streaming_port 43001 \
  --audio_source pyaudio \
  --respeaker_index 10 \
  --respeaker_rate 16000 \
  --sample_rate 16000 \
  --respeaker_channels 6 \
  --respeaker_mic_channels 1,2,3,4 \
  --chunk_seconds 4 \
  --skip_asr_policy silence \
  --raw_input_gain 3 \
  --save_raw_chunks \
  --save_enhanced \
  --vad_th 0.7

  
python Models/SimulStreaming/simulstreaming_whisper_server.py \
  --host localhost \
  --port 43001 \
  --language en \
  --task transcribe \
  --model_path ./small.pt \
  --min-chunk-size 2.0 \
  --audio_max_len 5 \
  --max_context_tokens 0 \
  --log-level WARNING 2>&1 | python Models/SimulStreaming/clean_transcript.py --doa-jsonl robot/results/dominant/live_asr_doa_latest.jsonl

python robot/go2w_live_asr_doa_input.py eth0

python robot/go2w_live_asr_doa_demo.py