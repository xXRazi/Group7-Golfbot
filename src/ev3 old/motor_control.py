#!/usr/bin/env python3
"""
motor_control.py  -  EV3 motor controller with turn-and-drive GOTO.

Written for Python 3.5 (ev3dev) - NO f-strings anywhere in this file.
All string formatting uses .format() or % so it runs on the EV3 brick.
"""

import math
import os
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent


def _load_degrees_per_turn_degree(default=2.35):
    """Load DEGREES_PER_TURN_DEGREE from calibration.txt if it exists.

    The file is written by calibrate_turn.py on the PC and must be copied
    to the same directory as this file on the EV3 SD card.
    If missing, the hard-coded default is used and a warning is printed.
    """
    calib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "calibration.txt")
    if os.path.exists(calib_path):
        try:
            value = float(open(calib_path).read().strip())
            print("[motor_control] Loaded DEGREES_PER_TURN_DEGREE={:.4f} from {}".format(
                value, calib_path))
            return value
        except (ValueError, OSError) as exc:
            print("[motor_control] WARNING: Could not read {}: {}. Using default {}.".format(
                calib_path, exc, default))
    else:
        print("[motor_control] WARNING: No calibration.txt found at {}. "
              "Using default DEGREES_PER_TURN_DEGREE={}. "
              "Run calibrate_turn.py on the PC to generate it.".format(
                  calib_path, default))
    return default


class MotorController:
    DEFAULT_TURN_SPEED  = 20
    DEFAULT_DRIVE_SPEED = 30

    MAP_COLS = 640
    MAP_ROWS = 480

    CM_PER_MAP_UNIT = 0.26

    # Loaded from calibration.txt - do NOT hard-code this.
    # Run calibrate_turn.py on the PC whenever wheels are changed,
    # then copy the resulting calibration.txt to the EV3.
    DEGREES_PER_TURN_DEGREE = _load_degrees_per_turn_degree(default=2.35)

    DEGREES_PER_CM       = 20
    DEGREES_PER_MAP_UNIT = DEGREES_PER_CM * CM_PER_MAP_UNIT

    LEFT_MOTOR_DIRECTION  = -1
    RIGHT_MOTOR_DIRECTION = -1

    TURN_BRAKE  = False
    DRIVE_BRAKE = False

    def __init__(self, left_output=OUTPUT_A, right_output=OUTPUT_D):
        self.tank = MoveTank(left_output, right_output)

        self.left_trim  = 0.0
        self.right_trim = 0.0

        self.x       = float(self.MAP_COLS) / 2.0
        self.y       = float(self.MAP_ROWS) / 2.0
        self.heading = 0.0

    # ── Utility ───────────────────────────────────────────────────────────────

    def clamp_speed(self, value):
        return max(-100.0, min(100.0, float(value)))

    def apply_trim(self, speed, trim):
        speed, trim = float(speed), float(trim)
        if speed > 0.0:
            return self.clamp_speed(speed + trim)
        if speed < 0.0:
            return self.clamp_speed(speed - trim)
        return 0.0

    def apply_calibration(self, left, right):
        return (self.apply_trim(left, self.left_trim),
                self.apply_trim(right, self.right_trim))

    def apply_motor_direction(self, left, right):
        return (self.clamp_speed(float(left)  * self.LEFT_MOTOR_DIRECTION),
                self.clamp_speed(float(right) * self.RIGHT_MOTOR_DIRECTION))

    def normalize_heading(self, angle):
        return float(angle) % 360.0

    def normalize_turn(self, angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def position_is_valid(self, x, y):
        return (0.0 <= float(x) < float(self.MAP_COLS) and
                0.0 <= float(y) < float(self.MAP_ROWS))

    def set_map_dimensions(self, rows, cols):
        rows = int(rows)
        cols = int(cols)

        if rows <= 0 or cols <= 0:
            print("MAP DIMENSIONS ERROR: rows={}, cols={}".format(rows, cols))
            return

        self.MAP_ROWS = rows
        self.MAP_COLS = cols
        print("MAP DIMENSIONS set to cols={}, rows={}".format(self.MAP_COLS, self.MAP_ROWS))

    # ── State setters (called by POSSYNC) ─────────────────────────────────────

    def set_heading(self, heading):
        self.heading = self.normalize_heading(heading)
        print("HEADING set to {:.2f}".format(self.heading))

    def set_position(self, x, y):
        x, y = float(x), float(y)
        if not self.position_is_valid(x, y):
            print("POSITION ERROR: ({:.1f}, {:.1f}) is outside map bounds "
                  "(x: 0..{}, y: 0..{})".format(
                      x, y, self.MAP_COLS - 1, self.MAP_ROWS - 1))
            return
        self.x = x
        self.y = y
        print("POSITION set to x={:.2f}, y={:.2f}".format(self.x, self.y))

    # ── Hardware actions ──────────────────────────────────────────────────────

    def stop(self, brake=True):
        self.tank.off(brake=brake)

    def calibrate(self, left_trim, right_trim):
        self.left_trim  = self.clamp_speed(left_trim)
        self.right_trim = self.clamp_speed(right_trim)
        print("CALIBRATE  left_trim={:.2f}  right_trim={:.2f}".format(
            self.left_trim, self.right_trim))

    def turn(self, angle, speed=0):
        """Rotate in place by angle degrees and update heading.

        angle > 0 : left (counter-clockwise)
        angle < 0 : right (clockwise)
        speed == 0: use DEFAULT_TURN_SPEED
        """
        angle = float(angle)
        if angle == 0.0:
            self.stop()
            return

        speed = abs(self.clamp_speed(float(speed) if speed != 0.0
                                     else float(self.DEFAULT_TURN_SPEED)))
        motor_degrees = abs(angle) * float(self.DEGREES_PER_TURN_DEGREE)

        print("TURN  requested={:.2f}  speed={:.1f}%  motor_degrees={:.2f}".format(
            angle, speed, motor_degrees))

        left_cmd  = -speed if angle > 0.0 else  speed
        right_cmd =  speed if angle > 0.0 else -speed

        left_cmd,  right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd,  right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=self.TURN_BRAKE,
            block=True,
        )

        self.heading = self.normalize_heading(self.heading + angle)
        print("HEADING now {:.2f}".format(self.heading))

    def drive_straight(self, distance_map_units, speed=0):
        """Drive straight distance_map_units map steps and update position.

        Positive distance = forward; negative = backward.
        """
        distance_map_units = float(distance_map_units)
        if distance_map_units == 0.0:
            self.stop()
            return

        speed = abs(self.clamp_speed(float(speed) if speed != 0.0
                                     else float(self.DEFAULT_DRIVE_SPEED)))
        distance_cm   = distance_map_units * float(self.CM_PER_MAP_UNIT)
        motor_degrees = abs(distance_map_units) * float(self.DEGREES_PER_MAP_UNIT)

        left_cmd  = speed  if distance_map_units > 0.0 else -speed
        right_cmd = speed  if distance_map_units > 0.0 else -speed

        left_cmd,  right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd,  right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        print("DRIVE  dist_units={:.2f}  dist_cm={:.2f}  "
              "speed={:.1f}%  heading={:.2f}  motor_degrees={:.2f}".format(
                  distance_map_units, distance_cm,
                  speed, self.heading, motor_degrees))

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=self.DRIVE_BRAKE,
            block=True,
        )

        heading_rad = math.radians(self.heading)
        self.x     += distance_map_units * math.cos(heading_rad)
        self.y     += distance_map_units * math.sin(heading_rad)

        print("POSITION now x={:.2f}, y={:.2f}".format(self.x, self.y))

    def goto(self, target_x, target_y, turn_speed=0, drive_speed=0):
        """Turn toward (target_x, target_y) then drive straight to it.

        target_x = column  (EV3 x)
        target_y = row     (EV3 y)
        """
        target_x = float(target_x)
        target_y = float(target_y)

        if not self.position_is_valid(target_x, target_y):
            print("GOTO ERROR: ({:.1f}, {:.1f}) is outside map bounds "
                  "(x: 0..{}, y: 0..{})".format(
                      target_x, target_y,
                      self.MAP_COLS - 1, self.MAP_ROWS - 1))
            return

        dx = target_x - self.x
        dy = target_y - self.y

        if dx == 0.0 and dy == 0.0:
            print("GOTO: already at target ({:.1f}, {:.1f})".format(
                target_x, target_y))
            return

        target_heading = self.normalize_heading(math.degrees(math.atan2(dy, dx)))
        turn_angle     = self.normalize_turn(target_heading - self.heading)
        distance       = math.hypot(dx, dy)

        print("GOTO  target=({:.1f}, {:.1f})  current=({:.1f}, {:.1f})  "
              "heading={:.1f}  target_heading={:.1f}  "
              "turn={:.1f}  dist_units={:.1f}  dist_cm={:.1f}".format(
                  target_x, target_y, self.x, self.y,
                  self.heading, target_heading,
                  turn_angle, distance,
                  distance * self.CM_PER_MAP_UNIT))

        self.turn(turn_angle, turn_speed)
        self.drive_straight(distance, drive_speed)

        self.x = target_x
        self.y = target_y
        print("POSITION snapped to x={:.2f}, y={:.2f}".format(self.x, self.y))

    def set_speed(self, left, right):
        left  = self.clamp_speed(left)
        right = self.clamp_speed(right)

        left,  right = self.apply_calibration(left, right)
        left,  right = self.apply_motor_direction(left, right)

        print("SET SPEED  left={:.2f}  right={:.2f}".format(left, right))

        if left == 0.0 and right == 0.0:
            self.stop()
            return

        self.tank.on(SpeedPercent(left), SpeedPercent(right))
