import time
import sys
from dataclasses import dataclass

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient

@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="damp", id=0),         
    TestOption(name="stand up", id=1),     
    # TestOption(name="stand down", id=2),   
    TestOption(name="stop move", id=3),    
    # TestOption(name="recovery stand", id=4),     
    TestOption(name="balance stand", id=5),
    TestOption(name="forward", id=6),
    TestOption(name="backward", id=7),
    TestOption(name="left", id=8),
    TestOption(name="right", id=9),
    TestOption(name="turn left", id=10),
    TestOption(name="turn right", id=11),
    # TestOption(name="speed low", id=12),
    # TestOption(name="speed medium", id=13),
    # TestOption(name="lean forward", id=14),
    # TestOption(name="lean back", id=15),
    # TestOption(name="level body", id=16),
    # TestOption(name="dance 1", id=17),
    # TestOption(name="dance 2", id=18),
    # TestOption(name="sit", id=19),
]

MOVE_SPEED = 0.3
SIDE_SPEED = 0.5
TURN_SPEED = 0.5
PITCH_ANGLE = 0.15

class UserInterface:
    def __init__(self):
        self.test_option_ = None

    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        input_str = input("Enter id or name: \n").strip().lower().replace("_", " ")

        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return

        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        self.test_option_.name = None
        self.test_option_.id = None
        print("No matching test option found.")

def execute_option(sport_client, test_option):
    code = None

    if test_option.id == 0:
        code = sport_client.Damp()
    elif test_option.id == 1:
        code = sport_client.StandUp()
    # elif test_option.id == 2:
    #     code = sport_client.StandDown()
    elif test_option.id == 3:
        code = sport_client.StopMove()
    # elif test_option.id == 4:
    #     code = sport_client.RecoveryStand()
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
    # elif test_option.id == 12:
    #     code = sport_client.SpeedLevel(0)
    # elif test_option.id == 13:
    #     code = sport_client.SpeedLevel(1)
    # elif test_option.id == 14:
    #     code = sport_client.Euler(0, PITCH_ANGLE, 0)
    # elif test_option.id == 15:
    #     code = sport_client.Euler(0, -PITCH_ANGLE, 0)
    # elif test_option.id == 16:
    #     code = sport_client.Euler(0, 0, 0)
    # elif test_option.id == 17:
    #     code = sport_client.Dance1()
    # elif test_option.id == 18:
    #     code = sport_client.Dance2()
    # elif test_option.id == 19:
    #     code = sport_client.Sit()

    print(f"Return code: {code}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = SportClient() 
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    while True:
        user_interface.terminal_handle()

        print(f"Updated Test Option: Name = {test_option.name}, ID = {test_option.id}\n")

        if test_option.id is not None:
            execute_option(sport_client, test_option)

        time.sleep(1)
