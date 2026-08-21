import time
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

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

COMMAND_FILE_NAME = "command_input.txt"
COMMAND_INTERVAL_SECONDS = 4
MOVE_SPEED = 0.3
SIDE_SPEED = 0.5
TURN_SPEED = 0.5


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


def get_command_file_path():
    if len(sys.argv) >= 3:
        command_file_path = Path(sys.argv[2])
        if not command_file_path.is_absolute():
            command_file_path = Path(__file__).resolve().parent / command_file_path
        return command_file_path

    return Path(__file__).resolve().parent / COMMAND_FILE_NAME

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface [commandTxtFile]")
        sys.exit(-1)

    command_file_path = get_command_file_path()
    if not command_file_path.exists():
        print(f"Command file not found: {command_file_path}")
        print(f"Create {COMMAND_FILE_NAME} in the same folder, or pass a txt file path as the second argument.")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    sport_client = SportClient() 
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    print(f"Reading commands from: {command_file_path}")
    print(f"Command interval: {COMMAND_INTERVAL_SECONDS}s\n")

    with command_file_path.open("r", encoding="utf-8") as command_file:
        for line_number, line in enumerate(command_file, start=1):
            line = line.strip()
            test_option = find_option_in_line(line)

            if test_option is None:
                print(f"Line {line_number}: no matching command found. Text: {line}")
            else:
                print(f"Line {line_number}: Test: {test_option.name}, test_id: {test_option.id}")
                execute_option(sport_client, test_option)

            time.sleep(COMMAND_INTERVAL_SECONDS)
