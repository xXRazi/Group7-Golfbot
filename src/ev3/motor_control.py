#!/usr/bin/env python3
import math
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent


class MotorController:
    DEFAULT_TURN_SPEED = 20
    DEFAULT_DRIVE_SPEED = 30

    # ------------------------------------------------------------
    # MAP SETTINGS
    # ------------------------------------------------------------

    # Your map is 480 rows by 640 columns.
    # Use:
    #   x = column, valid range 0..639
    #   y = row,    valid range 0..479
    MAP_ROWS = 480
    MAP_COLS = 640

    # Each map coordinate increment in centimeters.
    CM_PER_MAP_UNIT = 0.26

    # ------------------------------------------------------------
    # MOTOR CALIBRATION
    # ------------------------------------------------------------

    # Turn calibration.
    # This is motor degrees per requested robot turn degree.
    DEGREES_PER_TURN_DEGREE = 2.31

    # Wheel rotation per real-world cm.
    DEGREES_PER_CM = 17.6

    # Derived value:
    # motor degrees per one map coordinate increment.
    DEGREES_PER_MAP_UNIT = DEGREES_PER_CM * CM_PER_MAP_UNIT

    # Your robot drove backwards, so both motors are inverted.
    LEFT_MOTOR_DIRECTION = -1
    RIGHT_MOTOR_DIRECTION = -1

    # Turning with brake=False may reduce jerk/skid at the end of turns.
    TURN_BRAKE = False
    DRIVE_BRAKE = False

    def __init__(self, left_output=OUTPUT_A, right_output=OUTPUT_D):
        self.tank = MoveTank(left_output, right_output)

        self.left_trim = 0.0
        self.right_trim = 0.0

        # Robot pose in MAP UNITS, not centimeters.
        self.x = self.MAP_COLS / 2.0
        self.y = self.MAP_ROWS / 2.0
        self.heading = 0.0

        # Heading convention:
        #
        #   0 degrees   = facing +X
        #   90 degrees  = facing +Y
        #   180 degrees = facing -X
        #   270 degrees = facing -Y
        #
        # Positive turn = counterclockwise / left.
        #
        # This matches Python's math.atan2() and the x/y update math.

    def clamp_speed(self, value):
        """Clamp motor speed to valid EV3 percentage range."""
        value = float(value)

        if value < -100.0:
            return -100.0
        if value > 100.0:
            return 100.0
        return value

    def apply_trim(self, speed, trim):
        """
        Apply trim so that positive trim increases motor power
        in both directions.
        """
        speed = float(speed)
        trim = float(trim)

        if speed > 0.0:
            return self.clamp_speed(speed + trim)
        if speed < 0.0:
            return self.clamp_speed(speed - trim)
        return 0.0

    def apply_calibration(self, left, right):
        """Apply saved calibration trims to left/right motor speeds."""
        left = self.apply_trim(left, self.left_trim)
        right = self.apply_trim(right, self.right_trim)
        return left, right

    def apply_motor_direction(self, left, right):
        """
        Convert logical robot motor directions into physical EV3 motor directions.

        Because the robot drove backwards, both motor directions are inverted.
        """
        left = self.clamp_speed(float(left) * self.LEFT_MOTOR_DIRECTION)
        right = self.clamp_speed(float(right) * self.RIGHT_MOTOR_DIRECTION)
        return left, right

    def stop(self, brake=True):
        self.tank.off(brake=brake)

    def calibrate(self, left_trim, right_trim):
        self.left_trim = self.clamp_speed(left_trim)
        self.right_trim = self.clamp_speed(right_trim)

        print(
            "CALIBRATE left_trim={:.2f}, right_trim={:.2f}".format(
                self.left_trim,
                self.right_trim,
            )
        )

    def normalize_heading(self, angle):
        """Keep heading in range [0, 360)."""
        return float(angle) % 360.0

    def normalize_turn(self, angle):
        """
        Normalize angle to shortest turn in range [-180, 180).

        Positive = left / counterclockwise
        Negative = right / clockwise
        """
        angle = float(angle)
        return (angle + 180.0) % 360.0 - 180.0

    def position_is_valid(self, x, y):
        x = float(x)
        y = float(y)
        return 0.0 <= x < float(self.MAP_COLS) and 0.0 <= y < float(self.MAP_ROWS)

    def set_heading(self, heading):
        """Manually set the robot heading."""
        self.heading = self.normalize_heading(heading)
        print("HEADING set to {:.2f}".format(self.heading))

    def set_position(self, x, y):
        """Manually set the robot position in map coordinates."""
        x = float(x)
        y = float(y)

        if not self.position_is_valid(x, y):
            print(
                "POSITION ERROR: ({:.2f}, {:.2f}) is outside map bounds. "
                "Expected x=0..{}, y=0..{}.".format(
                    x,
                    y,
                    self.MAP_COLS - 1,
                    self.MAP_ROWS - 1,
                )
            )
            return

        self.x = x
        self.y = y
        print("POSITION set to x={:.2f}, y={:.2f}".format(self.x, self.y))

    def turn(self, angle, speed=0):
        """
        Turn in place and update heading.

        angle > 0 : left / counterclockwise
        angle < 0 : right / clockwise
        speed == 0: use default turn speed

        This function keeps angle, speed, and motor_degrees as floats.
        """
        angle = float(angle)
        speed = float(speed)

        if angle == 0.0:
            self.stop()
            return

        if speed == 0.0:
            speed = float(self.DEFAULT_TURN_SPEED)

        speed = abs(self.clamp_speed(speed))
        motor_degrees = abs(angle) * float(self.DEGREES_PER_TURN_DEGREE)

        print(
            "TURN requested_angle={:.4f}, speed={:.4f}, "
            "degrees_per_turn_degree={:.4f}, motor_degrees={:.4f}".format(
                angle,
                speed,
                float(self.DEGREES_PER_TURN_DEGREE),
                motor_degrees,
            )
        )

        if angle > 0.0:
            left_cmd = -speed
            right_cmd = speed
        else:
            left_cmd = speed
            right_cmd = -speed

        left_cmd, right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd, right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        print(
            "TURN motor_cmd left={:.4f}, right={:.4f}, brake={}".format(
                left_cmd,
                right_cmd,
                self.TURN_BRAKE,
            )
        )

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
        """
        Drive straight and update x/y based on current heading.

        distance_map_units is in map coordinates, not centimeters.
        Positive distance = forward
        Negative distance = backward
        """
        distance_map_units = float(distance_map_units)
        speed = float(speed)

        if distance_map_units == 0.0:
            self.stop()
            return

        if speed == 0.0:
            speed = float(self.DEFAULT_DRIVE_SPEED)

        speed = abs(self.clamp_speed(speed))

        distance_cm = distance_map_units * float(self.CM_PER_MAP_UNIT)
        motor_degrees = abs(distance_map_units) * float(self.DEGREES_PER_MAP_UNIT)

        if distance_map_units > 0.0:
            left_cmd = speed
            right_cmd = speed
        else:
            left_cmd = -speed
            right_cmd = -speed

        left_cmd, right_cmd = self.apply_calibration(left_cmd, right_cmd)
        left_cmd, right_cmd = self.apply_motor_direction(left_cmd, right_cmd)

        print(
            "DRIVE distance_map_units={:.4f}, distance_cm={:.4f}, "
            "speed={:.4f}, heading={:.4f}, motor_degrees={:.4f}".format(
                distance_map_units,
                distance_cm,
                speed,
                self.heading,
                motor_degrees,
            )
        )

        print(
            "DRIVE motor_cmd left={:.4f}, right={:.4f}, brake={}".format(
                left_cmd,
                right_cmd,
                self.DRIVE_BRAKE,
            )
        )

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=self.DRIVE_BRAKE,
            block=True,
        )

        heading_rad = math.radians(self.heading)
        self.x += distance_map_units * math.cos(heading_rad)
        self.y += distance_map_units * math.sin(heading_rad)

        print("POSITION now x={:.2f}, y={:.2f}".format(self.x, self.y))

    def goto(self, target_x, target_y, turn_speed=0, drive_speed=0):
        """
        Turn toward the target coordinate, then drive straight to it.

        target_x and target_y are map coordinates.
        """
        target_x = float(target_x)
        target_y = float(target_y)

        if not self.position_is_valid(target_x, target_y):
            print(
                "GOTO ERROR: ({:.2f}, {:.2f}) is outside map bounds. "
                "Expected x=0..{}, y=0..{}.".format(
                    target_x,
                    target_y,
                    self.MAP_COLS - 1,
                    self.MAP_ROWS - 1,
                )
            )
            return

        dx = target_x - self.x
        dy = target_y - self.y

        if dx == 0.0 and dy == 0.0:
            print("GOTO already at target x={:.2f}, y={:.2f}".format(target_x, target_y))
            return

        target_heading = math.degrees(math.atan2(dy, dx))
        target_heading = self.normalize_heading(target_heading)

        turn_angle = self.normalize_turn(target_heading - self.heading)
        distance_map_units = math.hypot(dx, dy)

        print(
            "GOTO target=({:.2f}, {:.2f}) current=({:.2f}, {:.2f}) "
            "current_heading={:.2f} target_heading={:.2f} "
            "turn_angle={:.2f} distance_map_units={:.2f} distance_cm={:.2f}".format(
                target_x,
                target_y,
                self.x,
                self.y,
                self.heading,
                target_heading,
                turn_angle,
                distance_map_units,
                distance_map_units * float(self.CM_PER_MAP_UNIT),
            )
        )

        self.turn(turn_angle, turn_speed)
        self.drive_straight(distance_map_units, drive_speed)

        # Snap the internal pose to the intended map coordinate.
        # This avoids small floating point drift after many goto commands.
        self.x = target_x
        self.y = target_y
        print("POSITION snapped to x={:.2f}, y={:.2f}".format(self.x, self.y))

    def set_speed(self, left, right):
        left = self.clamp_speed(left)
        right = self.clamp_speed(right)

        left, right = self.apply_calibration(left, right)
        left, right = self.apply_motor_direction(left, right)

        print("SET SPEED left={:.4f}, right={:.4f}".format(left, right))

        if left == 0.0 and right == 0.0:
            self.stop()
            return

        self.tank.on(SpeedPercent(left), SpeedPercent(right))