# Online Python Files

This folder contains online-style streaming evaluation scripts for the
IPDNet2 -> DSENet -> SimulStreaming Whisper pipeline. The scripts process saved
multichannel WAV files but send the selected enhanced audio to a streaming
Whisper server over a socket instead of calling local Whisper directly.

## File Descriptions

| File | Description |
| --- | --- |
| `pipeline_streaming.py` | Streams full-scene microphone data through the online-style pipeline. For each scene it estimates DOAs, enhances candidate DOAs, selects the loudest enhanced signal, sends it to SimulStreaming Whisper, and records timing, transcript, and DOA error outputs. |
| `pipeline_streaming_4schunks.py` | Streaming timing benchmark for the 4-second dataset variant. It follows the same SSL -> DSE -> streaming ASR flow as `pipeline_streaming.py`, but writes `_4s` result files and is aimed at pre-chunked audio data. No ground truth text is provided here in this dataset, so NO WER |
| `pipeline_streaming_4schunks_full.py` | Full 4-second chunk streaming evaluation. It groups microphone files by scene and chunk index, streams each selected enhanced chunk, assigns transcript segments to chunk intervals, reconstructs scene transcripts, and computes scene/corpus WER including a cleaned-text WER variant. |
| `pipeline_streaming_4schunks_full_noenhanced.py` | Direct-mixture streaming ASR baseline for the full 4-second chunk dataset. It skips SSL and DSENet, converts each multichannel mixture chunk to mono, streams it to SimulStreaming Whisper, reconstructs scene transcripts, and computes scene/corpus WER for comparison with the enhanced pipeline. Results default to `online/results/pipeline_streaming_4schunks_full_noenhanced/`. |
| `pipeline_streaming_4schunks_full_noenhanced_clean.py` | Direct-clean-speech streaming ASR baseline for the full 4-second chunk dataset. It reads `Eval/clean_4s` spk1 chunks, streams them directly to SimulStreaming Whisper, reconstructs scene transcripts, and computes scene/corpus WER as a clean-speech reference point. Results default to `online/results/pipeline_streaming_4schunks_full_noenhanced_clean/`. |

## Common Inputs

- Enhanced-pipeline scripts expect model checkpoints under `Models/SSL`,
  `Models/DSE`, and `Models/SimulStreaming`; the no-enhancement baseline only
  needs the SimulStreaming Whisper model.
- The scripts can either start a managed SimulStreaming Whisper server or
  connect to an externally running server.
- CLI arguments control dataset paths, model/checkpoint paths, streaming host
  and port, packet timing, realtime pacing, device, and output folder.

## Common Outputs

- Transcript JSONL files capture segments received from the streaming server.
- Detail CSV files contain per-scene or per-chunk timing, transcript fields,
  and, where applicable, selected DOA/enhanced-output fields.
- Summary JSON files report aggregate timing, realtime, transcript, DOA error,
  and, for the full 4-second script, WER metrics.
