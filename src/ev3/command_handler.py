#!/usr/bin/env python3
import struct

ERROR = 0x0
CALIBRATE = 0x1
SENDMAP = 0x2
HANDSHAKE = 0xA
GOTO = 0xB
POSSYNC = 0xC
TURN = 0xD
SETSPEED = 0xE
FINISH = 0xF
OPEN_CLAW = 0x10
CLOSE_CLAW = 0x11
DELIVER_BALL = 0x12

# Packet formats:
#
# ERROR:     [CMD]
# CALIBRATE: [CMD][LEFT_TRIM:int8][RIGHT_TRIM:int8]
# SENDMAP:   [CMD][ROWS:uint16][COLS:uint16][MAP_BYTES...]
# HANDSHAKE: [CMD]
# GOTO:      [CMD][X:int32][Y:int32]
# POSSYNC:   [CMD][X:int32][Y:int32]
# TURN:      [CMD][ANGLE:int16][SPEED:int8]
# SETSPEED:  [CMD][LEFT:int8][RIGHT:int8]
# FINISH:    [CMD]

ERROR_LENGTH = 1
CALIBRATE_LENGTH = 3
SENDMAP_HEADER_LENGTH = 5
HANDSHAKE_LENGTH = 1
GOTO_LENGTH = 9
POSSYNC_LENGTH = 11
TURN_LENGTH = 4
SETSPEED_LENGTH = 3
FINISH_LENGTH = 1
OPEN_CLAW_LENGTH = 1
CLOSE_CLAW_LENGTH = 1
DELIVER_BALL_LENGTH = 1


def byte_to_signed(value):
    """Convert one unsigned byte 0..255 to signed -128..127."""
    if value > 127:
        return value - 256
    return value


def read_int32(data, offset):
    return struct.unpack(">i", data[offset:offset + 4])[0]


def read_int16(data, offset):
    return struct.unpack(">h", data[offset:offset + 2])[0]


def read_uint16(data, offset):
    return struct.unpack(">H", data[offset:offset + 2])[0]


def sendmap_length(data):
    """
    SENDMAP format:
    [CMD][ROWS:uint16][COLS:uint16][MAP...]

    Total length = 5 + rows * cols.

    Returns None until enough header bytes are present to know the full size.
    """
    if len(data) < SENDMAP_HEADER_LENGTH:
        return None

    rows = read_uint16(data, 1)
    cols = read_uint16(data, 3)
    return SENDMAP_HEADER_LENGTH + (rows * cols)


class Command:
    def __init__(self, code, length, handler):
        self.code = code
        self.length = length
        self.handler = handler

    def get_expected_length(self, data):
        if callable(self.length):
            return self.length(data)
        return self.length

    def execute(self, data):
        expected_length = self.get_expected_length(data)

        if expected_length is None:
            print("Command 0x{:X} does not yet have enough header bytes".format(self.code))
            return False

        if len(data) < expected_length:
            print(
                "Invalid length for command 0x{:X}: expected at least {}, got {}".format(
                    self.code, expected_length, len(data)
                )
            )
            return False

        self.handler(data)
        return True


class CommandHandler:
    def __init__(self, motor_controller):
        self.motor_controller = motor_controller

        self.map = {
            "rows": 0,
            "cols": 0,
            "cells": [],
        }

        self.commands = {
            ERROR: Command(ERROR, ERROR_LENGTH, self.error_command),
            CALIBRATE: Command(CALIBRATE, CALIBRATE_LENGTH, self.calibrate_command),
            SENDMAP: Command(SENDMAP, sendmap_length, self.sendmap_command),
            HANDSHAKE: Command(HANDSHAKE, HANDSHAKE_LENGTH, self.handshake),
            GOTO: Command(GOTO, GOTO_LENGTH, self.goto),
            POSSYNC: Command(POSSYNC, POSSYNC_LENGTH, self.position_sync),
            TURN: Command(TURN, TURN_LENGTH, self.turn),
            SETSPEED: Command(SETSPEED, SETSPEED_LENGTH, self.set_speed),
            FINISH: Command(FINISH, FINISH_LENGTH, self.finish_command),
            OPEN_CLAW: Command(OPEN_CLAW, OPEN_CLAW_LENGTH, self.open_claw_command),
            CLOSE_CLAW: Command(CLOSE_CLAW, CLOSE_CLAW_LENGTH, self.close_claw_command),
            DELIVER_BALL: Command(DELIVER_BALL, DELIVER_BALL_LENGTH, self.deliver_ball_command),
        }

    def get_expected_length(self, data):
        if not data:
            return None

        cmd_code = data[0]
        command = self.commands.get(cmd_code)

        if command is None:
            return 1

        return command.get_expected_length(data)

    def handshake(self, data):
        print("HANDSHAKE")

    def goto(self, data):
        x = read_int32(data, 1)
        y = read_int32(data, 5)
        print("GO TO({}, {})".format(x, y))
        self.motor_controller.goto(x, y)

    def position_sync(self, data):
        x = read_int32(data, 1)
        y = read_int32(data, 5)
        heading = read_int16(data, 9) / 10.0      # tenths → degrees
        print("POSITION SYNCHRONIZATION to ({}, {}), heading={:.1f}".format(x, y, heading))
        self.motor_controller.set_position(x, y)
        self.motor_controller.set_heading(heading) # already exists in motor_control.py

    def turn(self, data):
        angle = read_int16(data, 1)
        speed = byte_to_signed(data[3])
        self.motor_controller.turn(angle, speed)

    def set_speed(self, data):
        left = byte_to_signed(data[1])
        right = byte_to_signed(data[2])
        self.motor_controller.set_speed(left, right)

    def error_command(self, data):
        print("ERROR COMMAND RECEIVED")
        self.motor_controller.stop()

    def calibrate_command(self, data):
        left_trim = byte_to_signed(data[1])
        right_trim = byte_to_signed(data[2])
        self.motor_controller.calibrate(left_trim, right_trim)

    def sendmap_command(self, data):
        rows = read_uint16(data, 1)
        cols = read_uint16(data, 3)

        expected_cells = rows * cols
        raw_cells = list(data[SENDMAP_HEADER_LENGTH:SENDMAP_HEADER_LENGTH + expected_cells])

        cells_2d = []
        for r in range(rows):
            start = r * cols
            end = start + cols
            cells_2d.append(raw_cells[start:end])

        self.map = {
            "rows": rows,
            "cols": cols,
            "cells": cells_2d,
        }

        print("SENDMAP rows={}, cols={}, cells={}".format(rows, cols, expected_cells))

    def finish_command(self, data):
        print("FINISH COMMAND")
        self.motor_controller.stop()

    def open_claw_command(self, data):
        print("OPEN CLAW COMMAND")
        self.motor_controller.open_claw()

    def close_claw_command(self, data):
        print("CLOSE CLAW COMMAND")
        self.motor_controller.close_claw()

    def deliver_ball_command(self, data):
        print("DELIVER BALL COMMAND")
        self.motor_controller.deliver_ball()

    def handle_command(self, data):
        if not data:
            return False

        cmd_code = data[0]
        command = self.commands.get(cmd_code)

        if command is None:
            print("Invalid command received: 0x{:X}".format(cmd_code))
            self.motor_controller.stop()
            return False

        return command.execute(data)