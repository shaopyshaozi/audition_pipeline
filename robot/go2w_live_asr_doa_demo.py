import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


COMMAND_INTERVAL_SECONDS = 4.0
MOVE_SPEED = 0.3
ROTATE_SPEED = 0.3
DEFAULT_LIVE_JSONL = Path(__file__).resolve().parent / "results" / "dominant" / "live_asr_doa_latest.jsonl"


@dataclass
class TestOption:
    name: Optional[str]
    id: Optional[int]


option_list = [
    TestOption(name="sit", id=0),
    TestOption(name="stand up", id=1),
    TestOption(name="forward", id=6),
    TestOption(name="backward", id=7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo-print Unitree Go2W actions from live ASR+DoA JSONL events.")
    parser.add_argument(
        "live_jsonl",
        nargs="?",
        type=Path,
        default=DEFAULT_LIVE_JSONL,
        help="Live ASR+DoA JSONL path from the robot pipeline.",
    )
    parser.add_argument(
        "--command-cooldown",
        type=float,
        default=COMMAND_INTERVAL_SECONDS,
        help="Minimum seconds between repeated command prints.",
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Only handle events marked is_final=true.",
    )
    parser.add_argument(
        "--allow-command-without-doa",
        action="store_true",
        help="Print text command even when selected_doa is missing or invalid.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between checks for new JSONL events.",
    )
    parser.add_argument(
        "--replay-existing",
        action="store_true",
        help="Start from the beginning of an existing JSONL file instead of only new events.",
    )
    return parser.parse_args()


def normalize_line(line):
    normalized_words = re.sub(r"[^a-z0-9]+", " ", line.lower()).split()
    return " ".join(normalized_words)


def find_option_in_line(line):
    normalized_line = normalize_line(line)

    if not normalized_line or normalized_line.startswith("#"):
        return None

    id_match = re.search(r"\bid\s*[:=]?\s*(\d+)\b", normalized_line)
    if id_match:
        input_id = int(id_match.group(1))
        for option in option_list:
            if option.id == input_id:
                return option

    sorted_options = sorted(option_list, key=lambda option: len(normalize_line(option.name)), reverse=True)

    for option in sorted_options:
        normalized_option = normalize_line(option.name)
        if f" {normalized_option} " in f" {normalized_line} ":
            return option

    return None


def demo_execute_option(test_option):
    if test_option.id == 0:
        print("DEMO ACTION: would call sport_client.Damp() for spoken command 'sit'.")
    elif test_option.id == 1:
        print("DEMO ACTION: would call sport_client.StandUp().")
    elif test_option.id == 6:
        print(f"DEMO ACTION: would call sport_client.Move({MOVE_SPEED}, 0, 0) for forward.")
    elif test_option.id == 7:
        print(f"DEMO ACTION: would call sport_client.Move({-MOVE_SPEED}, 0, 0) for backward.")


def demo_rotate_to_doa(speaker_doa):
    direction = "left/positive yaw" if speaker_doa >= 0 else "right/negative yaw"
    print(
        "DEMO ROTATE: would prepare BalanceStand, then rotate "
        f"{speaker_doa:.1f} deg toward the predicted DoA "
        f"using yaw speed {ROTATE_SPEED} ({direction})."
    )
    print("DEMO ROTATE: would stop rotation when target yaw is reached.")
    return True


def read_new_events(path: Path, offset: int) -> tuple[List[Dict], int]:
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


def should_skip_duplicate(
    event: Dict,
    option: TestOption,
    last_command_by_name: Dict[str, float],
    command_cooldown: float,
) -> bool:
    now = time.time()
    command_key = normalize_line(option.name or str(option.id))
    previous_time = last_command_by_name.get(command_key)
    if previous_time is not None and now - previous_time < command_cooldown:
        text = str(event.get("text", "")).strip()
        print(f"DEMO SKIP: repeated command within cooldown: {option.name} | text={text}")
        return True
    last_command_by_name[command_key] = now
    return False


def main() -> None:
    args = parse_args()

    print("DEMO MODE: no Unitree SDK commands will be sent.")
    print(f"Reading live ASR+DoA events from: {args.live_jsonl}")
    print(f"Command cooldown: {args.command_cooldown:.1f}s")
    print("Recognized commands: sit, stand up, forward, backward")

    offset = 0 if args.replay_existing else args.live_jsonl.stat().st_size if args.live_jsonl.exists() else 0
    seen_event_indices: Set[int] = set()
    last_command_by_name: Dict[str, float] = {}

    try:
        while True:
            events, offset = read_new_events(args.live_jsonl, offset)
            for event in events:
                if event.get("type") != "asr_doa":
                    continue
                event_index = event.get("event_index")
                if isinstance(event_index, int):
                    if event_index in seen_event_indices:
                        continue
                    seen_event_indices.add(event_index)
                if args.require_final and not bool(event.get("is_final", False)):
                    continue

                text = str(event.get("text", "")).strip()
                if not text:
                    continue

                test_option = find_option_in_line(text)
                if test_option is None:
                    print(f"DEMO IGNORE: no matching command. DoA={event.get('selected_doa')} text={text}")
                    continue
                if should_skip_duplicate(event, test_option, last_command_by_name, args.command_cooldown):
                    continue

                selected_doa = event.get("selected_doa")
                print(
                    f"DEMO COMMAND RECEIVED: command={test_option.name}, "
                    f"test_id={test_option.id}, DoA={selected_doa}, text={text}"
                )
                if selected_doa is None or float(selected_doa) < 0:
                    if not args.allow_command_without_doa:
                        print(f"DEMO SKIP: selected_doa is missing or invalid: {selected_doa}")
                        continue
                else:
                    demo_rotate_to_doa(float(selected_doa))
                demo_execute_option(test_option)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\nDEMO EXIT.")


if __name__ == "__main__":
    main()
