#!/usr/bin/env python3
import socket
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent

HOST = "0.0.0.0"
PORT = 5000

ERROR = 0x0
CALIBRATE = 0x1
HANDSHAKE = 0xA
GOTO = 0xB
POSSYNC = 0xC
TURN = 0xD
SETSPEED = 0xE
FINISH = 0xF

ERROR_LENGTH = 1          # [CMD]
CALIBRATE_LENGTH = 3      # [CMD][LEFT_TRIM][RIGHT_TRIM]
HANDSHAKE_LENGTH = 1      # [CMD]
GOTO_LENGTH = 3           # [CMD][X][Y]
POSSYNC_LENGTH = 3        # [CMD][X][Y]
TURN_LENGTH = 3           # [CMD][ANGLE][SPEED]
SETSPEED_LENGTH = 3       # [CMD][LEFT][RIGHT]
FINISH_LENGTH = 1         # [CMD]

# Motors
tank = MoveTank(OUTPUT_A, OUTPUT_D)

# Tuning values
DEFAULT_TURN_SPEED = 20
DEGREES_PER_TURN_UNIT = 5

# Speed values for motors. Values are set using calibration command.
left_trim = 30
right_trim = 30


class Command:
    def __init__(self, code, length, handler):
        self.code = code
        self.length = length
        self.handler = handler

    def execute(self, data):
        if len(data) < self.length:
            print(
                "Invalid length for command 0x{:X}: expected at least {}, got {}".format(
                    self.code, self.length, len(data)
                )
            )
            return False

        self.handler(data)
        return True


def byte_to_signed(value):
    """Convert one unsigned byte (0-255) to signed (-128 to 127)."""
    if value > 127:
        return value - 256
    return value


def clamp_speed(value):
    """Clamp motor speed to valid EV3 percentage range."""
    if value < -100:
        return -100
    if value > 100:
        return 100
    return value


def apply_trim(speed, trim):
    """
    Apply trim so that positive trim increases motor power in both directions.
    Example:
      speed=30, trim=5  -> 35
      speed=-30, trim=5 -> -35
    """
    if speed > 0:
        return clamp_speed(speed + trim)
    if speed < 0:
        return clamp_speed(speed - trim)
    return 0


def apply_calibration(left, right):
    """Apply saved calibration trims to left/right motor speeds."""
    left = apply_trim(left, left_trim)
    right = apply_trim(right, right_trim)
    return left, right


def stop_motors(brake=True):
    tank.off(brake=brake)


def handshake(data):
    print("HANDSHAKE")


def goto(data):
    x, y = data[1], data[2]
    print("GO TO({}, {})".format(x, y))
    # TODO implement pathfinding from current position
    # For now this only logs the target.


def position_sync(data):
    x, y = data[1], data[2]
    print("POSITION SYNCHRONIZATION to ({}, {})".format(x, y))
    # TODO use this for correction / navigation logic


def turn(data):
    angle = byte_to_signed(data[1])
    speed = byte_to_signed(data[2])

    if speed == 0:
        speed = DEFAULT_TURN_SPEED

    speed = abs(clamp_speed(speed))
    motor_degrees = abs(angle) * DEGREES_PER_TURN_UNIT

    print("TURN angle={}, speed={}".format(angle, speed))

    if angle == 0:
        stop_motors()
        return

    if angle > 0:
        # Turn right in place
        left_cmd = speed
        right_cmd = -speed
    else:
        # Turn left in place
        left_cmd = -speed
        right_cmd = speed

    left_cmd, right_cmd = apply_calibration(left_cmd, right_cmd)

    tank.on_for_degrees(
        SpeedPercent(left_cmd),
        SpeedPercent(right_cmd),
        motor_degrees,
        brake=True,
        block=True
    )


def set_speed(data):
    left = byte_to_signed(data[1])
    right = byte_to_signed(data[2])

    left = clamp_speed(left)
    right = clamp_speed(right)

    left, right = apply_calibration(left, right)

    print("SET SPEED left={}, right={}".format(left, right))

    if left == 0 and right == 0:
        stop_motors()
        return

    tank.on(SpeedPercent(left), SpeedPercent(right))


def error_command(data):
    print("ERROR COMMAND RECEIVED")
    stop_motors()


def calibrate_command(data):
    global left_trim, right_trim

    left_trim = clamp_speed(byte_to_signed(data[1]))
    right_trim = clamp_speed(byte_to_signed(data[2]))

    print("CALIBRATE left_trim={}, right_trim={}".format(left_trim, right_trim))


def finish_command(data):
    print("FINISH COMMAND")
    stop_motors()


COMMANDS = {
    ERROR: Command(ERROR, ERROR_LENGTH, error_command),
    CALIBRATE: Command(CALIBRATE, CALIBRATE_LENGTH, calibrate_command),
    HANDSHAKE: Command(HANDSHAKE, HANDSHAKE_LENGTH, handshake),
    GOTO: Command(GOTO, GOTO_LENGTH, goto),
    POSSYNC: Command(POSSYNC, POSSYNC_LENGTH, position_sync),
    TURN: Command(TURN, TURN_LENGTH, turn),
    SETSPEED: Command(SETSPEED, SETSPEED_LENGTH, set_speed),
    FINISH: Command(FINISH, FINISH_LENGTH, finish_command),
}


def handle_command(data):
    if not data:
        return False

    cmd_code = data[0]
    command = COMMANDS.get(cmd_code)

    if command is None:
        print("Invalid command received : 0x{:X}".format(cmd_code))
        stop_motors()
        return False

    return command.execute(data)


def loop(conn):
    while True:
        data_buf = conn.recv(16)

        if not data_buf:
            print("Client disconnected")
            stop_motors()
            break

        if not handle_command(data_buf):
            break

        print("Received raw bytes:", list(data_buf))

        response = b"EV3 got command\n"
        conn.sendall(response)


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(1)

        print("Listening on {}:{}...".format(HOST, PORT))

        conn, addr = server.accept()
        with conn:
            print("Connected by {}".format(addr))
            loop(conn)


if __name__ == "__main__":
    main()