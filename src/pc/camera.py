import cv2 as cv
import os
import socket
import tempfile
import time
import math
#from create_test_image import test_matrix
from Imagesplitter import create_matrix
from id_color import ball_pos_approx_shape, grapler_pos_approx, robot_pos, goals_pos_approx, robot_pose_approx
from dotenv import load_dotenv
from collection_algorithm import A_star, get_h_list
from com_protocol import (
    HOST,
    PORT,
    send_command,
    build_handshake,
    build_goto,
    build_possync,
    build_claw_open,
    build_claw_close,
    build_setspeed,
)
import numpy as np


# ==========================================
# 1. PERSPECTIVE WARP SETUP
# ==========================================

# Define your desired final grid size (640 columns by 480 rows)
width, height = 640, 360

# --- pts1: The Raw Camera Corners ---
# You still MUST measure these from your raw camera feed!
# I am using placeholder numbers here. If your physical arena isn't
# a perfect rectangle in the camera's eye, these numbers will not form a perfect box.
# order: top-left, top-right, bottom-right, bottom-left
pts1 = np.float32([
    [1, 0],       # top-left
    [1916, 1],     # top-right
    [1919, 1076],   # bottom-right
    [1, 1078]      # bottom-left
])

pts2 = np.float32([
    [0, 0],                 # top-left
    [width - 1, 0],         # top-right
    [width - 1, height - 1],# bottom-right
    [0, height - 1]         # bottom-left
])

# Compute the transformation matrix
warp_matrix = cv.getPerspectiveTransform(pts1, pts2)
# ==========================================
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
CAMERA_INDEX = 0
SYNC_DELAY_SECONDS = 0.2
SYNC_IMAGE_PATH = os.path.join(IMAGE_DIR, "robot_sync_frame.png")

# Stop this many warped-image/map units before the ball center.
# 35 units is about 9 cm with CM_PER_MAP_UNIT=0.26 on the EV3.
# Tune this value if the claw stops too far from or too close to the ball.
PICKUP_STOP_DISTANCE = 35

# Use smaller waypoints during pickup so the robot does not take one long final
# command that can push the ball away.
PICKUP_WAYPOINT_STEP_SIZE = 15

# Extra delay after forced stop, before closing the claw.
PICKUP_SETTLE_SECONDS = 0.15


def open_camera(camera_index=CAMERA_INDEX):
    print("using camera.py from : ", __file__)
    print("Trying to open camera : ", camera_index)

    camera = cv.VideoCapture(camera_index)

    if not camera.isOpened():
        print("Could not open camera index {}".format(camera_index))
        return None

    return camera


def close_camera(camera):
    if camera is not None:
        camera.release()
    cv.destroyAllWindows()


def warp_frame(frame):
    return cv.warpPerspective(frame, warp_matrix, (width, height))


def read_warped_frame(camera):
    res, frame = camera.read()

    if not res:
        print("Could not read camera frame")
        return None

    return warp_frame(frame)


def count_color(matrix, color):
    count = 0

    for row in matrix:
        for value in row:
            if value == color:
                count += 1

    return count


def get_robot_pose_from_camera(camera):
    warped_frame = read_warped_frame(camera)

    if warped_frame is None:
        return None

    cv.imwrite(SYNC_IMAGE_PATH, warped_frame)
    print("Saved sync image:", SYNC_IMAGE_PATH)

    color_matrix = create_matrix(SYNC_IMAGE_PATH)

    print(
        "Robot marker counts: Y={}, P={}, B={}".format(
            count_color(color_matrix, "Y"),
            count_color(color_matrix, "P"),
            count_color(color_matrix, "B"),
        )
    )

    return robot_pose_approx(color_matrix)


def show_camera_once(camera):
    warped_frame = read_warped_frame(camera)

    if warped_frame is None:
        return

    cv.imshow("camera", warped_frame)
    cv.waitKey(1)


def sync_robot_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return False

    x, y, heading = pose
    x = int(round(x))
    y = int(round(y))
    heading_tenths = int(round(heading * 10))

    print("Camera sync: x={}, y={}, heading={:.1f}".format(x, y, heading))

    return send_command(sock, build_possync(x, y, heading_tenths))


def goto_then_sync(sock, camera, row, col):
    x = int(round(col))
    y = int(round(row))

    print("Sending GOTO x={}, y={}".format(x, y))

    if not send_command(sock, build_goto(x, y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


def path_is_valid(robot_path):
    return robot_path and not isinstance(robot_path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def truncate_path_before_target(robot_path, stop_distance=PICKUP_STOP_DISTANCE):
    """
    Return a path that stops before the final target.

    The final A* point is normally the ball center. If the robot drives all the
    way to that point, the claw/robot can hit the ball and push it away. This
    walks backward from the final path point and chooses a pickup point
    stop_distance units before the ball.
    """
    if not path_is_valid(robot_path):
        print("Invalid path:", robot_path)
        return None

    if len(robot_path) < 2:
        return robot_path

    distance_from_target = 0.0

    for index in range(len(robot_path) - 1, 0, -1):
        distance_from_target += point_distance(robot_path[index], robot_path[index - 1])

        if distance_from_target >= stop_distance:
            pickup_path = robot_path[:index]

            if len(pickup_path) == 0:
                pickup_path = [robot_path[0]]

            print(
                "Pickup approach: ball={}, stop_point={}, stop_distance={:.1f}".format(
                    robot_path[-1],
                    pickup_path[-1],
                    distance_from_target,
                )
            )
            return pickup_path

    # If the path is very short, do not drive onto the ball. Stay at the start
    # point and close the claw there.
    print(
        "Pickup approach: path shorter than stop distance; staying at start {}".format(
            robot_path[0]
        )
    )
    return [robot_path[0]]


def follow_path_with_camera_sync(sock, camera, robot_path, step_size=10):
    if not path_is_valid(robot_path):
        print("Invalid path:", robot_path)
        return False

    if not sync_robot_from_camera(sock, camera):
        return False

    waypoints = robot_path[::step_size]

    if waypoints[-1] != robot_path[-1]:
        waypoints.append(robot_path[-1])

    for row, col in waypoints:
        if not goto_then_sync(sock, camera, row, col):
            return False

    return True


def approach_ball_and_close_claw(sock, camera, robot_path):
    """
    Open the claw, drive only to the pickup point before the ball, stop,
    then close the claw.
    """
    pickup_path = truncate_path_before_target(robot_path, PICKUP_STOP_DISTANCE)

    if not path_is_valid(pickup_path):
        return False

    print("Opening claw before pickup approach")
    if not send_command(sock, build_claw_open()):
        return False

    if not follow_path_with_camera_sync(
        sock,
        camera,
        pickup_path,
        step_size=PICKUP_WAYPOINT_STEP_SIZE,
    ):
        return False

    # Force the drive motors to brake before the claw closes. This matters
    # because the EV3 drive move may otherwise coast slightly at the end.
    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing claw at pickup point")
    return send_command(sock, build_claw_close())


def run_autonomous_camera():
    allocatedTime = 1
    STARTTIME = 2
    BeginTime = time.time()
    startTime = time.time()

    load_dotenv()
    path = IMAGE_DIR

    camera = open_camera(CAMERA_INDEX)
    if camera is None:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((HOST, PORT))
        send_command(sock, build_handshake())

        res, frame = camera.read()
        count = 0
        path_executed = False

        while camera.isOpened():
            res, frame = camera.read()

            if not res:
                continue

            warped_frame = warp_frame(frame)

            BeginElapsedTime = time.time() - BeginTime

            if BeginElapsedTime >= STARTTIME:

                elapsedTime = time.time() - startTime

                if elapsedTime >= allocatedTime:
                    elapsedTime = 0
                    startTime = time.time()

                    im_ = f"{count}.png"
                    full_path = os.path.join(path, im_)
                    cv.imwrite(full_path, warped_frame)
                    #Directory skal være hvor du har projektet gemt
                    count += 1
                    print("Vi tager et billede")
                    #color_matrix = create_matrix(full_path)

                    color_matrix = create_matrix(full_path)

                    white_list = ball_pos_approx_shape(color_matrix, "W")

                    grapler_point = grapler_pos_approx(color_matrix, "G")
                    print(grapler_point)

                    if grapler_point is None:
                        print("No grapler detected; cannot collect ball")
                        continue

                    if not white_list:
                        print("No white balls detected")
                        continue

                    min_list = []
                    for item in white_list:
                        value = get_h_list(grapler_point[0], grapler_point[1], item[0], item[1])
                        min_list.append(value)
                    print("minlist", min_list)
                    paired = list(zip(min_list, white_list))
                    paired.sort()  # sorts by min_list values
                    white_list = [item for _, item in paired]
                    print("white_list", white_list)
                    robot_path = A_star(color_matrix, grapler_point, white_list[0])
                    #robot_path = A_star(color_matrix, white_list[0], white_list[-1])

                    if not path_executed:
                        path_executed = approach_ball_and_close_claw(sock, camera, robot_path)

                    robot_position = robot_pos(color_matrix)
                    #print("robot_position", robot_position)

                    Goal_A, Goal_B = goals_pos_approx(color_matrix, "PK", "C")
                    print("Goal_A:", Goal_A)
                    print("Goal_B:", Goal_B)

                    orangeball_pos = ball_pos_approx_shape(color_matrix, "O")
                    print("orangeball_pos:", orangeball_pos)

            cv.imshow("camera", warped_frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        sock.close()
        close_camera(camera)


if __name__ == "__main__":
    run_autonomous_camera()