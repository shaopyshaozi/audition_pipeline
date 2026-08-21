## Non-dominant fixed-DoA test, 2 terminals needed, one for each command below
## Run from the repository root:
##   cd /home/shaozi/ucl/code/audition_pipeline

# Terminal 1: SimulStreaming Whisper server + live ASR/DoA display.
# This reads the stable JSONL written by the non-dominant pipeline.
python3 Models/SimulStreaming/simulstreaming_whisper_server.py \
  --host localhost \
  --port 43001 \
  --language en \
  --task transcribe \
  --model_path Models/SimulStreaming/small.pt \
  --min-chunk-size 2.0 \
  --audio_max_len 5 \
  --max_context_tokens 0 \
  --log-level WARNING 2>&1 | python3 Models/SimulStreaming/clean_transcript.py \
  --doa-jsonl robot/results/non_dominant/live_asr_doa_latest.jsonl

# Terminal 2: fixed-DoA non-dominant enhancement pipeline.
# Put the command speaker around DoA 0 degrees; other speakers can stay active at other directions.
python3 robot/pipeline_streaming_4schunks_full_respeaker_linux_non_dominant_2.py \
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
  --target_doa 0 \
  --skip_asr_policy silence \
  --raw_input_gain 3 \
  --save_raw_chunks \
  --save_enhanced

# Optional: find the correct PyAudio device index.
python3 robot/pipeline_streaming_4schunks_full_respeaker_linux_non_dominant_2.py --list_audio_devices

# Optional Terminal 3: robot command consumer using the non-dominant JSONL.
python3 robot/go2w_live_asr_doa_input.py eth0 robot/results/non_dominant/live_asr_doa_latest.jsonl

# Optional demo-only command consumer, no robot required.
python3 robot/go2w_live_asr_doa_demo.py robot/results/non_dominant/live_asr_doa_latest.jsonl
