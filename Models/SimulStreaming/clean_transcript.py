import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretty-print live Whisper output, optionally joined with DoA events.")
    parser.add_argument(
        "--doa-jsonl",
        type=Path,
        default=None,
        help="Live ASR+DoA JSONL file written by the robot pipeline.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between checks when tailing --doa-jsonl.",
    )
    parser.add_argument(
        "--show-raw-whisper",
        action="store_true",
        help="Also print raw Whisper text from stdin when --doa-jsonl is enabled.",
    )
    return parser.parse_args()


def whisper_text_from_line(line: str) -> Optional[Dict]:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    text = str(data.get("text", "")).strip()
    if not text:
        return None
    return data


def format_whisper(data: Dict) -> str:
    text = str(data.get("text", "")).strip()
    emission_time = data.get("emission_time")
    if emission_time is not None:
        return f"[emitted {float(emission_time):.2f}s] {text}"
    return text


def format_asr_doa(event: Dict) -> str:
    start = event.get("transcript_start_sec")
    end = event.get("transcript_end_sec")
    if start is None:
        start = event.get("audio_start_sec", 0.0)
    if end is None:
        end = event.get("audio_end_sec", start)
    doa = event.get("selected_doa")
    doa_label = "n/a" if doa is None or int(doa) < 0 else f"{int(doa):03d} deg"
    chunk_index = event.get("chunk_index", "?")
    text = str(event.get("text", "")).strip()
    return f"[{float(start):.2f}-{float(end):.2f}s | DoA {doa_label} | chunk {chunk_index}] {text}"


def read_new_jsonl_lines(path: Path, offset: int) -> tuple[List[Dict], int]:
    if not path.exists():
        return [], offset
    size = path.stat().st_size
    if offset > size:
        offset = 0
    events = []
    with path.open("r", encoding="utf-8") as file_obj:
        file_obj.seek(offset)
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        offset = file_obj.tell()
    return events, offset


def tail_doa_events(
    path: Path,
    stop_event: threading.Event,
    poll_interval: float,
    full_text: List[str],
    seen_events: Set[int],
) -> None:
    offset = path.stat().st_size if path.exists() else 0
    while not stop_event.is_set():
        events, offset = read_new_jsonl_lines(path, offset)
        for event in events:
            if event.get("type") != "asr_doa":
                continue
            event_index = event.get("event_index")
            if isinstance(event_index, int):
                if event_index in seen_events:
                    continue
                seen_events.add(event_index)
            text = str(event.get("text", "")).strip()
            if not text:
                continue
            print(format_asr_doa(event), flush=True)
            full_text.append(text)
        time.sleep(poll_interval)


def main() -> None:
    args = parse_args()
    full_text: List[str] = []
    stop_event = threading.Event()
    tail_thread: Optional[threading.Thread] = None
    seen_events: Set[int] = set()

    if args.doa_jsonl is not None:
        tail_thread = threading.Thread(
            target=tail_doa_events,
            args=(args.doa_jsonl, stop_event, args.poll_interval, full_text, seen_events),
            daemon=True,
        )
        tail_thread.start()

    try:
        for line in sys.stdin:
            data = whisper_text_from_line(line)
            if data is None:
                continue
            if args.doa_jsonl is None or args.show_raw_whisper:
                print(format_whisper(data), flush=True)
                full_text.append(str(data.get("text", "")).strip())
    finally:
        if tail_thread is not None:
            time.sleep(max(args.poll_interval * 2.0, 0.2))
            stop_event.set()
            tail_thread.join(timeout=1.0)

    print("\n========== FULL TRANSCRIPT ==========\n")
    print(" ".join(full_text))


if __name__ == "__main__":
    main()
