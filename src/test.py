#!/usr/bin/env python3

from ev3dev2.motor import MoveTank, MediumMotor, OUTPUT_A, OUTPUT_D, OUTPUT_B
from ev3dev2.sensor.lego import TouchSensor
from ev3dev2.sensor import INPUT_1, INPUT_2
from time import sleep

tank = MoveTank(OUTPUT_D, OUTPUT_A)
claw = MediumMotor(OUTPUT_B)

drive_button = TouchSensor(INPUT_1)
claw_button = TouchSensor(INPUT_2)

drive_direction = 1
claw_direction = 1

while True:

    # DRIVE CONTROL
    if drive_button.is_pressed:

        drive_direction *= -1

        tank.on(80 * drive_direction, 60 * drive_direction)

        while drive_button.is_pressed:
            sleep(0.01)

    else:
        tank.off(brake=True)


    # CLAW CONTROL
    if claw_button.is_pressed:

        claw_direction *= -1

        claw.on(60 * claw_direction)

        while claw_button.is_pressed:
            sleep(0.01)

    else:
        claw.off()


    sleep(0.05)
