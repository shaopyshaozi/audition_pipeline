import math
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.go2.sport.sport_client import SportClient


TOPIC_LOWSTATE = "rt/lowstate"
ROTATE_SPEED = 0.3
YAW_TOLERANCE_DEG = 1.0
CONTROL_INTERVAL = 0.05
MAX_ROTATE_TIME = 20.0


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

    print("Target reached. Stopping and running Dance1.")
    dance_code = sport_client.Dance1()
    print(f"Dance1 return code: {dance_code}")
    return True


def read_doa_from_terminal():
    while True:
        input_str = input("Enter speaker DoA in degrees, or q to quit: ").strip()

        if input_str.lower() in ("q", "quit", "exit"):
            return None

        try:
            return float(input_str)
        except ValueError:
            print("Invalid input. Please enter a number, for example: 55")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("Positive DoA rotates with Move(0, 0, +0.3); negative DoA rotates with Move(0, 0, -0.3).")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    yaw_reader = RobotYawReader()

    sport_client = SportClient()
    sport_client.SetTimeout(10.0)
    sport_client.Init()

    while True:
        speaker_doa = read_doa_from_terminal()
        if speaker_doa is None:
            sport_client.StopMove()
            print("Exit.")
            break

        rotate_to_doa(sport_client, yaw_reader, speaker_doa)
