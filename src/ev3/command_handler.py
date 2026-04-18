#!/usr/bin/env python3

ERROR = 0x0
CALIBRATE = 0x1
SENDMAP = 0x2
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


def byte_to_signed(value):
    """Convert one unsigned byte (0-255) to signed (-128 to 127)."""
    if value > 127:
        return value - 256
    return value


def sendmap_length(data):
    """
    SENDMAP format:
    [CMD][ROWS][COLS][MAP...]
    Total length = 3 + rows * cols
    """
    if len(data) < 3:
        return 3

    rows = data[1]
    cols = data[2]
    return 3 + (rows * cols)


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

        # Holds the latest map received
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
        }

    def handshake(self, data):
        print("HANDSHAKE")

    def goto(self, data):
        x, y = data[1], data[2]
        print("GO TO({}, {})".format(x, y))
        # TODO implement pathfinding from current position

    def position_sync(self, data):
        x, y = data[1], data[2]
        print("POSITION SYNCHRONIZATION to ({}, {})".format(x, y))
        # TODO use this for correction / navigation logic

    def turn(self, data):
        angle = byte_to_signed(data[1])
        speed = byte_to_signed(data[2])
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
        rows = data[1]
        cols = data[2]
        raw_cells = list(data[3:3 + rows * cols])

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

        print("SENDMAP rows={}, cols={}".format(rows, cols))
        print("MAP stored:", self.map)

    def finish_command(self, data):
        print("FINISH COMMAND")
        self.motor_controller.stop()

    def handle_command(self, data):
        if not data:
            return False

        cmd_code = data[0]
        command = self.commands.get(cmd_code)

        if command is None:
            print("Invalid command received : 0x{:X}".format(cmd_code))
            self.motor_controller.stop()
            return False

        return command.execute(data)