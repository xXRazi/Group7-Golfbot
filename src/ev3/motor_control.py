#!/usr/bin/env python3
from ev3dev2.motor import OUTPUT_A, OUTPUT_D, MoveTank, SpeedPercent


class MotorController:
    DEFAULT_TURN_SPEED = 20
    DEGREES_PER_TURN_UNIT = 5

    def __init__(self, left_output=OUTPUT_A, right_output=OUTPUT_D):
        self.tank = MoveTank(left_output, right_output)

        self.left_trim = 0
        self.right_trim = 0

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

    def turn(self, angle, speed):
        """
        Turn in place.
        angle > 0 : right
        angle < 0 : left
        speed == 0: use default turn speed
        """
        if speed == 0:
            speed = self.DEFAULT_TURN_SPEED

        speed = abs(self.clamp_speed(speed))
        motor_degrees = abs(angle) * self.DEGREES_PER_TURN_UNIT

        print("TURN angle={}, speed={}".format(angle, speed))

        if angle == 0:
            self.stop()
            return

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

    def set_speed(self, left, right):
        left = self.clamp_speed(left)
        right = self.clamp_speed(right)

        left, right = self.apply_calibration(left, right)

        print("SET SPEED left={}, right={}".format(left, right))

        if left == 0 and right == 0:
            self.stop()
            return

        self.tank.on(SpeedPercent(left), SpeedPercent(right))