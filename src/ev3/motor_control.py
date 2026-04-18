#!/usr/bin/env python3
import math
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent

class MotorController:
    DEFAULT_TURN_SPEED = 20
    DEFAULT_DRIVE_SPEED = 30

    # Calibration constants - tune these for your robot
    DEGREES_PER_TURN_DEGREE = 5       # motor degrees needed for 1 degree robot turn
    DEGREES_PER_CM = 20               # motor degrees needed to drive 1 cm forward

    def __init__(self, left_output=OUTPUT_A, right_output=OUTPUT_D):
        self.tank = MoveTank(left_output, right_output)

        self.left_trim = 0
        self.right_trim = 0

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        # Convention:
        # 0 degrees = facing +X direction
        # 90 degrees = facing +Y direction
        # Positive turn = clockwise/right

    def clamp_speed(self, value):
        """Clamp motor speed to valid EV3 percentage range."""
        if value < -100:
            return -100
        if value > 100:
            return 100
        return value

    def apply_trim(self, speed, trim):
        """
        Apply trim so that positive trim increases motor power
        in both directions.

        Example:
          speed=30, trim=5  -> 35
          speed=-30, trim=5 -> -35
        """
        if speed > 0:
            return self.clamp_speed(speed + trim)
        if speed < 0:
            return self.clamp_speed(speed - trim)
        return 0

    def apply_calibration(self, left, right):
        """Apply saved calibration trims to left/right motor speeds."""
        left = self.apply_trim(left, self.left_trim)
        right = self.apply_trim(right, self.right_trim)
        return left, right

    def stop(self, brake=True):
        self.tank.off(brake=brake)

    def calibrate(self, left_trim, right_trim):
        self.left_trim = self.clamp_speed(left_trim)
        self.right_trim = self.clamp_speed(right_trim)

        print(
            "CALIBRATE left_trim={}, right_trim={}".format(
                self.left_trim, self.right_trim
            )
        )

    def normalize_heading(self, angle):
        """Keep heading in range [0, 360)."""
        return angle % 360

    def normalize_turn(self, angle):
        """
        Normalize angle to shortest turn in range [-180, 180).
        Positive = right turn
        Negative = left turn
        """
        angle = (angle + 180) % 360 - 180
        return angle

    def set_heading(self, heading):
        """Manually set the robot heading."""
        self.heading = self.normalize_heading(heading)
        print("HEADING set to {:.1f}".format(self.heading))

    def set_position(self, x, y):
        """Manually set the robot position."""
        self.x = float(x)
        self.y = float(y)
        print("POSITION set to x={:.1f}, y={:.1f}".format(self.x, self.y))

    def turn(self, angle, speed=0):
        """
        Turn in place and update heading.
        angle > 0 : right
        angle < 0 : left
        speed == 0: use default turn speed
        """
        if angle == 0:
            self.stop()
            return

        if speed == 0:
            speed = self.DEFAULT_TURN_SPEED

        speed = abs(self.clamp_speed(speed))
        motor_degrees = abs(angle) * self.DEGREES_PER_TURN_DEGREE

        print("TURN angle={}, speed={}".format(angle, speed))

        if angle > 0:
            # Turn right in place
            left_cmd = speed
            right_cmd = -speed
        else:
            # Turn left in place
            left_cmd = -speed
            right_cmd = speed

        left_cmd, right_cmd = self.apply_calibration(left_cmd, right_cmd)

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=True,
            block=True,
        )

        # Update stored heading after turn
        self.heading = self.normalize_heading(self.heading + angle)
        print("HEADING now {:.1f}".format(self.heading))

    def drive_straight(self, distance_cm, speed=0):
        """
        Drive straight and update x/y based on current heading.
        Positive distance = forward
        Negative distance = backward
        """
        if distance_cm == 0:
            self.stop()
            return

        if speed == 0:
            speed = self.DEFAULT_DRIVE_SPEED

        speed = abs(self.clamp_speed(speed))
        motor_degrees = abs(distance_cm) * self.DEGREES_PER_CM

        if distance_cm > 0:
            left_cmd = speed
            right_cmd = speed
        else:
            left_cmd = -speed
            right_cmd = -speed

        left_cmd, right_cmd = self.apply_calibration(left_cmd, right_cmd)

        print(
            "DRIVE distance_cm={}, speed={}, heading={:.1f}".format(
                distance_cm, speed, self.heading
            )
        )

        self.tank.on_for_degrees(
            SpeedPercent(left_cmd),
            SpeedPercent(right_cmd),
            motor_degrees,
            brake=True,
            block=True,
        )

        # Update stored position after move
        heading_rad = math.radians(self.heading)
        self.x += distance_cm * math.cos(heading_rad)
        self.y += distance_cm * math.sin(heading_rad)

        print("POSITION now x={:.1f}, y={:.1f}".format(self.x, self.y))

    def goto(self, target_x, target_y, turn_speed=0, drive_speed=0):
        """
        Turn toward the target coordinate, then drive straight to it.
        """
        dx = target_x - self.x
        dy = target_y - self.y

        if dx == 0 and dy == 0:
            print("GOTO already at target x={}, y={}".format(target_x, target_y))
            return

        # atan2 gives angle from +X axis, matching our heading convention
        target_heading = math.degrees(math.atan2(dy, dx))
        target_heading = self.normalize_heading(target_heading)

        turn_angle = self.normalize_turn(target_heading - self.heading)
        distance_cm = math.hypot(dx, dy)

        print(
            "GOTO target=({:.1f}, {:.1f}) current=({:.1f}, {:.1f}) "
            "target_heading={:.1f} turn_angle={:.1f} distance={:.1f}".format(
                target_x, target_y,
                self.x, self.y,
                target_heading, turn_angle, distance_cm
            )
        )

        self.turn(turn_angle, turn_speed)
        self.drive_straight(distance_cm, drive_speed)

    def set_speed(self, left, right):
        left = self.clamp_speed(left)
        right = self.clamp_speed(right)

        left, right = self.apply_calibration(left, right)

        print("SET SPEED left={}, right={}".format(left, right))

        if left == 0 and right == 0:
            self.stop()
            return

        self.tank.on(SpeedPercent(left), SpeedPercent(right))