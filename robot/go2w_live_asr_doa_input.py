import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


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


option_list = [
    TestOption(name="damp", id=0),
    TestOption(name="stand up", id=1),
    TestOption(name="stop move", id=3),
    TestOption(name="balance stand", id=5),
    TestOption(name="forward", id=6),
    TestOption(name="backward", id=7),
    TestOption(name="left", id=8),
    TestOption(name="right", id=9),
    TestOption(name="turn left", id=10),
    TestOption(name="turn right", id=11),
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
        "--no-rotate-before-command",
        action="store_true",
        help="Execute text command without first rotating toward selected_doa.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="Seconds between checks for new JSONL events.",
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


def execute_option(sport_client, test_option):
    code = None

    if test_option.id == 0:
        code = sport_client.Damp()
    elif test_option.id == 1:
        code = sport_client.StandUp()
    elif test_option.id == 3:
        code = sport_client.StopMove()
    elif test_option.id == 5:
        code = sport_client.BalanceStand()
    elif test_option.id == 6:
        code = sport_client.Move(MOVE_SPEED, 0, 0)
    elif test_option.id == 7:
        code = sport_client.Move(-MOVE_SPEED, 0, 0)
    elif test_option.id == 8:
        code = sport_client.Move(0, SIDE_SPEED, 0)
    elif test_option.id == 9:
        code = sport_client.Move(0, -SIDE_SPEED, 0)
    elif test_option.id == 10:
        code = sport_client.Move(0, 0, TURN_SPEED)
    elif test_option.id == 11:
        code = sport_client.Move(0, 0, -TURN_SPEED)

    print(f"Return code: {code}")


def wrap_degrees(angle):
    return (angle + 180.0) % 360.0 - 180.0


def rotate_to_doa(sport_client, yaw_reader, speaker_doa):
    start_yaw = yaw_reader.wait_for_yaw()
    if start_yaw is None:
        print("No robot yaw received. Check the network interface and LowState topic.")
        return False

    print("Preparing BalanceStand before rotation.")
    balance_code = sport_client.BalanceStand()
    print(f"BalanceStand return code: {balance_code}")
    time.sleep(1.0)

    target_yaw = wrap_degrees(start_yaw + speaker_doa)
    start_time = time.time()

    print(f"Start yaw: {start_yaw:.1f} deg")
    print(f"Speaker DoA: {speaker_doa:.1f} deg")
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
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.network_interface)

    yaw_reader = RobotYawReader()

    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    offset = args.live_jsonl.stat().st_size if args.live_jsonl.exists() else 0
    seen_event_indices: Set[int] = set()
    last_command_by_name: Dict[str, float] = {}
    rotate_before_command = not args.no_rotate_before_command

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
                    print(f"No matching command. DoA={event.get('selected_doa')} text={text}")
                    continue
                if should_skip_duplicate(event, test_option, last_command_by_name, args.command_cooldown):
                    continue

                selected_doa = event.get("selected_doa")
                print(
                    f"Command: {test_option.name}, test_id: {test_option.id}, "
                    f"DoA: {selected_doa}, text: {text}"
                )
                if rotate_before_command and selected_doa is not None and float(selected_doa) >= 0:
                    rotate_to_doa(sport_client, yaw_reader, float(selected_doa))
                execute_option(sport_client, test_option)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        sport_client.StopMove()
        print("\nExit.")


if __name__ == "__main__":
    main()
