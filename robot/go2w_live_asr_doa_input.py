import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNITREE_ROOT = PROJECT_ROOT / "Models" / "unitree_sdk2_python"
if UNITREE_ROOT.is_dir():
    sys.path.insert(0, str(UNITREE_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


TOPIC_LOWSTATE = "rt/lowstate"
COMMAND_INTERVAL_SECONDS = 4.0
MOVE_SPEED = 0.3
SIDE_SPEED = 0.5
TURN_SPEED = 0.5
ROTATE_SPEED = 0.3
YAW_TOLERANCE_DEG = 1.0
CONTROL_INTERVAL = 0.05
MAX_ROTATE_TIME = 20.0
DEFAULT_LIVE_JSONL = Path(__file__).resolve().parent / "results" / "dominant" / "live_asr_doa_latest.jsonl"


@dataclass
class TestOption:
    name: Optional[str]
    id: Optional[int]
    aliases: Tuple[str, ...] = ()


option_list = [
    TestOption(name="sit", id=0),
    TestOption(name="stand up", id=1),
    TestOption(name="forward", id=6),
    TestOption(name="backward", id=7, aliases=("backwards",)),
]


class RobotYawReader:
    def __init__(self):
        self.latest_state = None
        self.subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.subscriber.Init(self.low_state_handler, 10)

    def low_state_handler(self, msg: LowState_):
        self.latest_state = msg

    def get_robot_yaw(self):
        if self.latest_state is None:
            return None
        return math.degrees(self.latest_state.imu_state.rpy[2])

    def wait_for_yaw(self, timeout=5.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            yaw = self.get_robot_yaw()
            if yaw is not None:
                return yaw
            time.sleep(CONTROL_INTERVAL)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive Unitree Go2W from live ASR+DoA JSONL events.")
    parser.add_argument("network_interface", help="Network interface used by the Unitree SDK, for example eth0.")
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
        help="Minimum seconds between repeated command executions.",
    )
    parser.add_argument(
        "--require-final",
        action="store_true",
        help="Only execute events marked is_final=true.",
    )
    parser.add_argument(
        "--allow-command-without-doa",
        action="store_true",
        help="Execute text command even when selected_doa is missing or invalid.",
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
    parser.add_argument(
        "--command-context-events",
        type=int,
        default=3,
        help="Number of recent ASR events used to detect split commands such as 'stand' + 'up'.",
    )
    parser.add_argument(
        "--command-context-sec",
        type=float,
        default=8.0,
        help="Maximum transcript time span used to detect split commands.",
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
        names = (option.name, *option.aliases)
        for name in names:
            normalized_option = normalize_line(name)
            if f" {normalized_option} " in f" {normalized_line} ":
                return option

    return None


def event_time_sec(event: Dict) -> float:
    for key in ("transcript_end_sec", "audio_end_sec", "received_wall_sec"):
        value = event.get(key)
        if value is not None:
            return float(value)
    return time.time()


def command_names(option: TestOption) -> Tuple[str, ...]:
    return (option.name or "", *option.aliases)


def command_words(option: TestOption) -> Set[str]:
    words: Set[str] = set()
    for name in command_names(option):
        words.update(normalize_line(name).split())
    return words


def find_option_with_recent_context(
    current_text: str,
    recent_texts: Sequence[str],
) -> Tuple[Optional[TestOption], str, bool]:
    direct = find_option_in_line(current_text)
    if direct is not None:
        return direct, current_text, False

    current_words = set(normalize_line(current_text).split())
    if not current_words:
        return None, current_text, False

    combined_text = " ".join([*recent_texts, current_text])
    normalized_combined = f" {normalize_line(combined_text)} "
    sorted_options = sorted(option_list, key=lambda option: len(normalize_line(option.name)), reverse=True)

    for option in sorted_options:
        if not (current_words & command_words(option)):
            continue
        for name in command_names(option):
            normalized_name = normalize_line(name)
            if f" {normalized_name} " in normalized_combined:
                return option, combined_text, True

    return None, current_text, False


def trim_recent_events(
    recent_events: List[Tuple[float, str]],
    now_sec: float,
    max_events: int,
    max_span_sec: float,
) -> List[Tuple[float, str]]:
    if max_events <= 0:
        return []
    trimmed = [
        (ts, text)
        for ts, text in recent_events
        if now_sec - ts <= max_span_sec
    ]
    return trimmed[-max_events:]


def prepare_balance_stand(sport_client):
    print("Preparing BalanceStand before rotation.")
    balance_code = sport_client.BalanceStand()
    print(f"BalanceStand return code: {balance_code}")
    time.sleep(1.0)


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def rotate_to_doa(sport_client, yaw_reader, speaker_doa):
    start_yaw = yaw_reader.wait_for_yaw()
    if start_yaw is None:
        print("No robot yaw received. Check the network interface and LowState topic.")
        return False

    turn_degrees = speaker_doa - 360 if speaker_doa > 180 else speaker_doa
    target_yaw = wrap_degrees(start_yaw + turn_degrees)
    start_time = time.time()

    print(f"Start yaw: {start_yaw:.1f} deg")
    print(f"Speaker DoA: {speaker_doa:.1f} deg")
    print(f"Turn degrees: {turn_degrees:.1f} deg")
    print(f"Target yaw: {target_yaw:.1f} deg")

    try:
        while True:
            current_yaw = yaw_reader.get_robot_yaw()
            if current_yaw is None:
                print("Lost robot yaw state.")
                return False

            error = wrap_degrees(target_yaw - current_yaw)
            print(f"Current yaw: {current_yaw:.1f} deg, error: {error:.1f} deg")

            if abs(error) < YAW_TOLERANCE_DEG:
                break

            if time.time() - start_time > MAX_ROTATE_TIME:
                print("Rotation timed out before reaching the target yaw.")
                return False

            if error > 0:
                sport_client.Move(0, 0, ROTATE_SPEED)
            else:
                sport_client.Move(0, 0, -ROTATE_SPEED)

            time.sleep(CONTROL_INTERVAL)
    finally:
        sport_client.StopMove()

    print("Target reached.")
    return True


def rotate_for_command(sport_client, yaw_reader, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    if speaker_doa is None:
        print("Skipping DoA rotation because selected_doa is missing or invalid.")
        return True

    rotated = rotate_to_doa(sport_client, yaw_reader, speaker_doa)
    if rotated:
        return True

    if allow_action_without_rotation:
        print("Continuing command because --allow-command-without-doa is set.")
        return True

    print("Skipping command because rotation to DoA failed.")
    return False


def execute_sit_command(sport_client, yaw_reader, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    prepare_balance_stand(sport_client)
    if not rotate_for_command(sport_client, yaw_reader, speaker_doa, allow_action_without_rotation):
        return False

    code = sport_client.Damp()
    print(f"Sit return code: {code}")
    return True


def execute_stand_up_command(sport_client, yaw_reader, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    print("Executing StandUp before DoA rotation.")
    stand_up_code = sport_client.StandUp()
    print(f"StandUp return code: {stand_up_code}")
    prepare_balance_stand(sport_client)
    return rotate_for_command(sport_client, yaw_reader, speaker_doa, allow_action_without_rotation)


def execute_forward_command(sport_client, yaw_reader, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    prepare_balance_stand(sport_client)
    if not rotate_for_command(sport_client, yaw_reader, speaker_doa, allow_action_without_rotation):
        return False

    code = sport_client.Move(MOVE_SPEED, 0, 0)
    print(f"Forward return code: {code}")
    return True


def execute_backward_command(sport_client, yaw_reader, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    prepare_balance_stand(sport_client)
    if not rotate_for_command(sport_client, yaw_reader, speaker_doa, allow_action_without_rotation):
        return False

    code = sport_client.Move(-MOVE_SPEED, 0, 0)
    print(f"Backward return code: {code}")
    return True


def execute_command(sport_client, yaw_reader, test_option, speaker_doa: Optional[float], allow_action_without_rotation: bool):
    command_handlers = {
        0: execute_sit_command,
        1: execute_stand_up_command,
        6: execute_forward_command,
        7: execute_backward_command,
    }
    handler = command_handlers.get(test_option.id)
    if handler is None:
        print(f"No command handler configured for test_id={test_option.id}.")
        return False
    return handler(sport_client, yaw_reader, speaker_doa, allow_action_without_rotation)


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
        print(f"Skipping repeated command within cooldown: {option.name} | text={text}")
        return True
    last_command_by_name[command_key] = now
    return False


def main() -> None:
    args = parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print(f"Reading live ASR+DoA events from: {args.live_jsonl}")
    print(f"Command cooldown: {args.command_cooldown:.1f}s")
    print(
        "Split-command context: "
        f"{args.command_context_events} events, {args.command_context_sec:g}s"
    )
    print("Recognized commands: sit, stand up, forward, backward")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.network_interface)

    yaw_reader = RobotYawReader()

    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    offset = 0 if args.replay_existing else args.live_jsonl.stat().st_size if args.live_jsonl.exists() else 0
    seen_event_indices: Set[int] = set()
    last_command_by_name: Dict[str, float] = {}
    recent_events: List[Tuple[float, str]] = []
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

                now_sec = event_time_sec(event)
                recent_events = trim_recent_events(
                    recent_events,
                    now_sec=now_sec,
                    max_events=args.command_context_events,
                    max_span_sec=args.command_context_sec,
                )
                recent_texts = [recent_text for _, recent_text in recent_events]
                test_option, matched_text, used_context = find_option_with_recent_context(text, recent_texts)
                recent_events.append((now_sec, text))

                if test_option is None:
                    continue
                if should_skip_duplicate(event, test_option, last_command_by_name, args.command_cooldown):
                    continue

                selected_doa = event.get("selected_doa")
                print(
                    f"\nCOMMAND RECEIVED: command={test_option.name}, "
                    f"test_id={test_option.id}, DoA={selected_doa}, "
                    f"used_context={used_context}, text={text}"
                )
                if used_context:
                    print(f"CONTEXT MATCH: {matched_text}")
                speaker_doa: Optional[float] = None
                if selected_doa is not None:
                    try:
                        speaker_doa = float(selected_doa)
                    except (TypeError, ValueError):
                        speaker_doa = None

                if speaker_doa is None or speaker_doa < 0:
                    if not args.allow_command_without_doa:
                        print(f"Skipping command because selected_doa is missing or invalid: {selected_doa}")
                        continue
                    speaker_doa = None

                execute_command(
                    sport_client,
                    yaw_reader,
                    test_option,
                    speaker_doa,
                    args.allow_command_without_doa,
                )

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        sport_client.StopMove()
        print("\nExit.")


if __name__ == "__main__":
    main()
