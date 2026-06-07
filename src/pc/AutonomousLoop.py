import cv2 as cv
import os
import time
import socket
import math

from dotenv import load_dotenv
from Imagesplitter import create_matrix
from id_color import ball_pos_approx_shape, grapler_pos_approx, goals_pos_approx
from collection_algorithm import A_star
from com_protocol import (
    HOST,
    PORT,
    send_command,
    build_handshake,
    build_goto,
    build_finish,
    build_setspeed,
    build_claw_open,
    build_claw_close,
    build_claw_deliver,
)

allocatedTime = 1
STARTTIME = 2

# Stop this many warped-image/map units before the ball center.
# 35 units is about 9 cm with CM_PER_MAP_UNIT=0.26 on the EV3.
PICKUP_STOP_DISTANCE = 35

# Smaller pickup waypoints prevent one long final movement from pushing the ball.
PICKUP_WAYPOINT_STEP_SIZE = 15
PICKUP_SETTLE_SECONDS = 0.15

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "images")

camera = cv.VideoCapture(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
send_command(sock, build_handshake())


def open_claw(sock):
    return send_command(sock, build_claw_open())


def close_claw(sock):
    return send_command(sock, build_claw_close())


def deliver_ball(sock):
    return send_command(sock, build_claw_deliver())


def closest_point(origin, points):
    if origin is None or not points:
        return None

    return min(
        points,
        key=lambda p: ((origin[0] - p[0]) ** 2 + (origin[1] - p[1]) ** 2) ** 0.5
    )


def path_is_valid(path):
    return path and not isinstance(path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def truncate_path_before_target(path, stop_distance=PICKUP_STOP_DISTANCE):
    """Return a path that stops before the final target/ball center."""
    if not path_is_valid(path):
        print("Invalid path:", path)
        return None

    if len(path) < 2:
        return path

    distance_from_target = 0.0

    for index in range(len(path) - 1, 0, -1):
        distance_from_target += point_distance(path[index], path[index - 1])

        if distance_from_target >= stop_distance:
            pickup_path = path[:index]

            if len(pickup_path) == 0:
                pickup_path = [path[0]]

            print(
                "Pickup approach: ball={}, stop_point={}, stop_distance={:.1f}".format(
                    path[-1],
                    pickup_path[-1],
                    distance_from_target,
                )
            )
            return pickup_path

    print("Pickup approach: path shorter than stop distance; staying at start {}".format(path[0]))
    return [path[0]]


def follow_path(sock, path, step_size=40):
    if not path_is_valid(path):
        print("Invalid path:", path)
        return False

    waypoints = path[::step_size]

    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])

    for row, col in waypoints:
        x = int(col)
        y = int(row)

        print("Goto:", x, y)

        if not send_command(sock, build_goto(x, y)):
            return False

        time.sleep(0.2)

    return True


def follow_pickup_path_and_close(sock, path):
    pickup_path = truncate_path_before_target(path, PICKUP_STOP_DISTANCE)

    if not path_is_valid(pickup_path):
        return False

    if not follow_path(sock, pickup_path, step_size=PICKUP_WAYPOINT_STEP_SIZE):
        return False

    # Force the drive motors to brake before the claw closes.
    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing grapler")
    return close_claw(sock)


def take_picture_and_matrix(camera, count):
    res, frame = camera.read()

    if not res:
        return None, None

    im_ = f"{count}.png"
    full_path = os.path.join(path, im_)
    cv.imwrite(full_path, frame)

    color_matrix = create_matrix(full_path)

    return frame, color_matrix


def collect_ball(color_matrix, sock, camera, count, target, goal):
    grapler_point = grapler_pos_approx(color_matrix, "G")

    if grapler_point is None:
        print("Could not find grapler")
        return False, count

    if not open_claw(sock):
        return False, count

    path_to_ball = A_star(color_matrix, grapler_point, target)

    if not follow_pickup_path_and_close(sock, path_to_ball):
        return False, count

    time.sleep(0.5)  # give robot/claw time to settle

    # Take new image after pickup
    new_frame, new_color_matrix = take_picture_and_matrix(camera, count)
    count += 1

    if new_color_matrix is None:
        print("Could not take new image after pickup")
        return False, count

    new_grapler_point = grapler_pos_approx(new_color_matrix, "G")

    if new_grapler_point is None:
        print("Could not re-detect grapler after pickup")
        return False, count

    # Re-detect goal too, in case camera/image changed
    new_goals = goals_pos_approx(new_color_matrix, "PK", "C")

    if new_goals is None:
        print("Could not re-detect goals after pickup")
        return False, count

    Goal_A, Goal_B = new_goals

    # Keep using the same chosen goal type
    updated_goal = goal

    print("New grapler position:", new_grapler_point)
    print("Driving from new grapler position to goal:", updated_goal)

    path_to_goal = A_star(new_color_matrix, new_grapler_point, updated_goal)

    if not follow_path(sock, path_to_goal):
        return False, count

    print("Shooting")
    if not deliver_ball(sock):
        return False, count

    return True, count


count = 0
BeginTime = time.time()
startTime = time.time()

try:
    while camera.isOpened():
        res, preview_frame = camera.read()

        if res:
            cv.imshow("camera", preview_frame)

        if time.time() - BeginTime < STARTTIME:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if time.time() - startTime >= allocatedTime:
            startTime = time.time()

            frame, color_matrix = take_picture_and_matrix(camera, count)
            count += 1

            if color_matrix is None:
                continue

            print("Image taken")

            grapler_point = grapler_pos_approx(color_matrix, "G")

            if grapler_point is None:
                print("No grapler detected")
                continue

            white_balls = ball_pos_approx_shape(color_matrix, "W")
            orange_balls = ball_pos_approx_shape(color_matrix, "O")

            goals = goals_pos_approx(color_matrix, "PK", "C")

            if goals is None:
                print("Could not find goals")
                continue

            Goal_A, Goal_B = goals

            # Priority: collect orange first if visible
            if orange_balls:
                target = closest_point(grapler_point, orange_balls)
                goal = Goal_A
                print("Collecting orange ball:", target)

            elif white_balls:
                target = closest_point(grapler_point, white_balls)
                goal = Goal_B
                print("Collecting white ball:", target)

            else:
                print("No balls left")
                send_command(sock, build_finish())
                break

            success, count = collect_ball(color_matrix, sock, camera, count, target, goal)

            if not success:
                print("Collection failed, trying again on next frame")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    sock.close()
    camera.release()
    cv.destroyAllWindows()