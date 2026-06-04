source /mnt/d/邵鹏远/UCL/博1/code/FN-SSL/IPDnet2/venv_wsl/bin/activate

# server side
python simulstreaming_whisper_server.py \
  --host localhost \
  --port 43001 \
  --language en \
  --task transcribe \
  --model_path ./small.pt \
  --min-chunk-size 1.0 \
  --audio_max_len 30 \
  --log-level WARNING 2>&1 | python clean_transcript.py

# sender side
python local_sending_enhanced.py \
  --host localhost \
  --port 43001 \
  --realtime