#!/usr/bin/env python3
import cv2 as cv
import os
import socket
import sys
import tempfile
import time

import numpy as np

from Imagesplitter import create_matrix
from id_color import robot_pose_approx
from com_protocol import (
    HOST,
    PORT,
    build_calibrate,
    build_finish,
    build_goto,
    build_handshake,
    build_possync,
    build_setspeed,
    build_turn,
    send_command,
)


# ==========================================
# PERSPECTIVE WARP SETUP
# Keep these values the same as camera.py.
# ==========================================

width, height = 640, 480

pts1 = np.float32([
    [1, 0],      # Top-Left
    [636, 1],    # Top-Right
    [639, 476],  # Bottom-Left
    [1, 478]     # Bottom-Right
])

pts2 = np.float32([
    [0, 0],
    [width, 0],
    [0, height],
    [width, height]
])

warp_matrix = cv.getPerspectiveTransform(pts1, pts2)

# ==========================================

CAMERA_INDEX = 1
SYNC_DELAY_SECONDS = 0.2
SYNC_IMAGE_PATH = os.path.join(tempfile.gettempdir(), "robot_sync_frame.png")


def get_robot_pose_from_camera(camera):
    res, frame = camera.read()

    if not res:
        print("Could not read camera frame")
        return None

    warped_frame = cv.warpPerspective(frame, warp_matrix, (width, height))
    cv.imwrite(SYNC_IMAGE_PATH, warped_frame)

    color_matrix = create_matrix(SYNC_IMAGE_PATH)
    return robot_pose_approx(color_matrix)


def sync_robot_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return True

    x, y, heading = pose

    x = int(round(x))
    y = int(round(y))
    heading_tenths = int(round(heading * 10))

    print("Camera sync: x={}, y={}, heading={:.1f}".format(x, y, heading))

    return send_command(sock, build_possync(x, y, heading_tenths))


def goto_xy_then_sync(sock, camera, x, y):
    x = int(x)
    y = int(y)

    print("Sending GOTO x={}, y={}".format(x, y))

    if not send_command(sock, build_goto(x, y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


def show_camera_once(camera):
    res, frame = camera.read()

    if not res:
        print("Could not read camera frame")
        return

    warped_frame = cv.warpPerspective(frame, warp_matrix, (width, height))
    cv.imshow("camera", warped_frame)
    cv.waitKey(1)


def print_help():
    print()
    print("Commands:")
    print("  goto X Y              drive to X,Y, then sync from camera automatically")
    print("  sync                  sync EV3 position from camera now")
    print("  raw_goto X Y          drive to X,Y without camera sync")
    print("  possync X Y HEADING   manually sync pose, heading in tenths of a degree")
    print("  turn ANGLE SPEED")
    print("  setspeed LEFT RIGHT")
    print("  stop")
    print("  calibrate LEFT_TRIM RIGHT_TRIM")
    print("  handshake")
    print("  preview               show one warped camera frame")
    print("  finish")
    print("  help")
    print("  quit")
    print()
    print("Notes:")
    print("  goto uses EV3 coordinates directly: x = column, y = row")
    print("  after each goto, this program captures a warped camera frame and sends POSSYNC")
    print()


def interactive_loop(sock, camera):
    print_help()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "goto":
                if len(parts) != 3:
                    print("Usage: goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                if not goto_xy_then_sync(sock, camera, x, y):
                    break

            elif cmd == "sync":
                if not sync_robot_from_camera(sock, camera):
                    break

            elif cmd == "raw_goto":
                if len(parts) != 3:
                    print("Usage: raw_goto X Y")
                    continue

                x = int(parts[1])
                y = int(parts[2])

                if not send_command(sock, build_goto(x, y)):
                    break

            elif cmd == "possync":
                if len(parts) != 4:
                    print("Usage: possync X Y HEADING_TENTHS")
                    continue

                x = int(parts[1])
                y = int(parts[2])
                heading_tenths = int(parts[3])

                if not send_command(sock, build_possync(x, y, heading_tenths)):
                    break

            elif cmd == "turn":
                if len(parts) != 3:
                    print("Usage: turn ANGLE SPEED")
                    continue

                angle = int(parts[1])
                speed = int(parts[2])

                if not send_command(sock, build_turn(angle, speed)):
                    break

            elif cmd == "setspeed":
                if len(parts) != 3:
                    print("Usage: setspeed LEFT RIGHT")
                    continue

                left = int(parts[1])
                right = int(parts[2])

                if not send_command(sock, build_setspeed(left, right)):
                    break

            elif cmd == "stop":
                if not send_command(sock, build_setspeed(0, 0)):
                    break

            elif cmd == "calibrate":
                if len(parts) != 3:
                    print("Usage: calibrate LEFT_TRIM RIGHT_TRIM")
                    continue

                left_trim = int(parts[1])
                right_trim = int(parts[2])

                if not send_command(sock, build_calibrate(left_trim, right_trim)):
                    break

            elif cmd == "handshake":
                if not send_command(sock, build_handshake()):
                    break

            elif cmd == "preview":
                show_camera_once(camera)

            elif cmd == "finish":
                if not send_command(sock, build_finish()):
                    break

            elif cmd == "help":
                print_help()

            elif cmd in ("quit", "exit"):
                break

            else:
                print("Unknown command. Type 'help'.")

        except ValueError as exc:
            print("Invalid input:", exc)


def main():
    host = HOST
    port = PORT

    if len(sys.argv) >= 2:
        host = sys.argv[1]

    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    camera = cv.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Could not open camera index {}".format(CAMERA_INDEX))
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))

            print("Connected to EV3 at {}:{}".format(host, port))

            if not send_command(sock, build_handshake()):
                return

            interactive_loop(sock, camera)

    except OSError as exc:
        print("Connection error:", exc)

    finally:
        camera.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()