## TCP connections, 3 terminals needed, one for each command below------------------------------------------------------------------------------

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


python pipeline_streaming_4schunks_full_respeaker.py \
  --audio_source tcp \
  --tcp_host 0.0.0.0 \
  --tcp_port 50007 \
  --skip_asr_policy silence

# skip_asr_policy can be silence (sending silence to asr), mixture (sending raw/not enhanced to asr), drop (skip) depending on use case

python record.py --list

python record.py --send_tcp --device_index 10 --capture_rate 16000 --rate 16000 --tcp_host 127.0.0.1 --tcp_port 50007 --save_chunks


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