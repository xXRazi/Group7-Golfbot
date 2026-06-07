import cv2 as cv
import math
import os
import socket
import time

import numpy as np
from dotenv import load_dotenv

from Imagesplitter import create_matrix
from id_color import (
    ball_pos_approx_shape,
    goals_pos_approx,
    grapler_pos_approx,
    robot_pos,
    robot_pose_approx,
)
from com_protocol import (
    HOST,
    PORT,
    build_claw_close,
    build_claw_open,
    build_goto,
    build_handshake,
    build_possync,
    build_setspeed,
    build_turn,
    send_command,
)


# Perspective-corrected map size.
width, height = 640, 360

pts1 = np.float32([
    [1, 0],
    [1916, 1],
    [1919, 1076],
    [1, 1078],
])
pts2 = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1],
])
warp_matrix = cv.getPerspectiveTransform(pts1, pts2)

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

CAMERA_INDEX = 0
SYNC_DELAY_SECONDS = 0.25
SYNC_IMAGE_PATH = os.path.join(IMAGE_DIR, "robot_sync_frame.png")
PICKUP_RECHECK_IMAGE_PATH = os.path.join(IMAGE_DIR, "pickup_recheck_frame.png")

# GOTO moves the ROBOT CENTER, not the green grappler point.
DEFAULT_GRAPPLER_OFFSET_DISTANCE = 42.0
MIN_GRAPPLER_OFFSET_DISTANCE = 20.0
MAX_GRAPPLER_OFFSET_DISTANCE = 85.0

# Pickup tuning.
# If the robot still touches/pushes the ball before closing, raise this to 14-16.
# If it stops too far away, lower it to 8-10.
PICKUP_GRAPPLER_STOP_SHORT = 12.0
PICKUP_READY_TO_CLOSE_DISTANCE = 30.0
PICKUP_HEADING_TOLERANCE = 5.0
PICKUP_MAX_TURN_ATTEMPTS = 5
PICKUP_MAX_APPROACH_STEPS = 10
PICKUP_MAX_FORWARD_STEP = 26.0
PICKUP_MIN_FORWARD_STEP = 5.0
PICKUP_TARGET_MARGIN = 8.0
MAX_TARGET_REUSE_DISTANCE = 90.0

BALL_CONFIRM_FRAMES = 2
BALL_CONFIRM_MAX_MOVE = 18.0
BALL_CONFIRM_DELAY_SECONDS = 0.08
CLAW_OPEN_WAIT_SECONDS = 0.45
PICKUP_SETTLE_SECONDS = 0.20
FINAL_TURN_SPEED = 15


def open_camera(camera_index=CAMERA_INDEX):
    print("using camera.py from:", __file__)
    print("Trying to open camera:", camera_index)
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
    return sum(row.count(color) for row in matrix)


def capture_color_state(camera, image_path):
    warped_frame = read_warped_frame(camera)
    if warped_frame is None:
        return None

    cv.imwrite(image_path, warped_frame)
    print("Saved camera image:", image_path)
    color_matrix = create_matrix(image_path)

    return {
        "frame": warped_frame,
        "matrix": color_matrix,
        "pose": robot_pose_approx(color_matrix),
        "grappler": grapler_pos_approx(color_matrix, "G"),
    }


def get_robot_pose_from_camera(camera):
    state = capture_color_state(camera, SYNC_IMAGE_PATH)
    if state is None:
        return None

    color_matrix = state["matrix"]
    print(
        "Robot marker counts: Y={}, P={}, B={}".format(
            count_color(color_matrix, "Y"),
            count_color(color_matrix, "P"),
            count_color(color_matrix, "B"),
        )
    )
    return state["pose"]


def show_camera_once(camera):
    warped_frame = read_warped_frame(camera)
    if warped_frame is not None:
        cv.imshow("camera", warped_frame)
        cv.waitKey(1)


def path_is_valid(robot_path):
    return robot_path and not isinstance(robot_path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def row_col_to_xy(point):
    return float(point[1]), float(point[0])


def xy_to_row_col(x, y):
    return int(round(y)), int(round(x))


def pose_center_xy(pose):
    return float(pose[0]), float(pose[1])


def normalize_heading(angle):
    return float(angle) % 360.0


def normalize_turn(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def clamp_xy_to_map(x, y, margin=PICKUP_TARGET_MARGIN):
    x = max(margin, min(width - 1 - margin, float(x)))
    y = max(margin, min(height - 1 - margin, float(y)))
    return x, y


def closest_point(origin, points):
    if origin is None or not points:
        return None
    return min(points, key=lambda point: point_distance(origin, point))


def choose_pickup_ball(color_matrix, ball_color, reference_point, target_hint=None):
    balls = ball_pos_approx_shape(color_matrix, ball_color)
    if not balls:
        print("No {} balls visible".format(ball_color))
        return None

    if target_hint is not None:
        hinted = closest_point(target_hint, balls)
        if hinted is not None and point_distance(target_hint, hinted) <= MAX_TARGET_REUSE_DISTANCE:
            chosen = hinted
        else:
            chosen = closest_point(reference_point, balls)
    else:
        chosen = closest_point(reference_point, balls)

    print("Chosen {} ball: {} from candidates {}".format(ball_color, chosen, balls))
    return chosen


def estimate_grappler_offset_distance(pose, grappler_point):
    if pose is not None and grappler_point is not None:
        center_x, center_y = pose_center_xy(pose)
        grappler_x, grappler_y = row_col_to_xy(grappler_point)
        distance = math.hypot(grappler_x - center_x, grappler_y - center_y)
        if MIN_GRAPPLER_OFFSET_DISTANCE <= distance <= MAX_GRAPPLER_OFFSET_DISTANCE:
            print("Live grappler offset distance: {:.1f}".format(distance))
            return distance
        print(
            "Ignoring unlikely grappler offset {:.1f}; using default {:.1f}".format(
                distance,
                DEFAULT_GRAPPLER_OFFSET_DISTANCE,
            )
        )
    return DEFAULT_GRAPPLER_OFFSET_DISTANCE


def heading_from_pose_to_ball(pose, ball_point):
    center_x, center_y = pose_center_xy(pose)
    ball_x, ball_y = row_col_to_xy(ball_point)
    return normalize_heading(math.degrees(math.atan2(ball_y - center_y, ball_x - center_x)))


def ball_reference_from_state(state):
    pose = state["pose"]
    grappler_point = state["grappler"]
    if grappler_point is not None:
        return grappler_point
    if pose is not None:
        center_x, center_y = pose_center_xy(pose)
        return xy_to_row_col(center_x, center_y)
    return None


def get_pickup_state(camera, ball_color, target_hint=None):
    state = capture_color_state(camera, PICKUP_RECHECK_IMAGE_PATH)
    if state is None:
        return None, None

    if state["pose"] is None:
        print("Could not detect robot pose for pickup")
        return state, None

    reference_point = ball_reference_from_state(state)
    ball_point = choose_pickup_ball(state["matrix"], ball_color, reference_point, target_hint)
    return state, ball_point


def send_pose_sync(sock, pose):
    if pose is None:
        print("Cannot sync: robot pose is None")
        return False

    x, y, heading = pose
    x = int(round(x))
    y = int(round(y))
    heading_tenths = int(round(normalize_heading(heading) * 10))
    print("Camera sync: x={}, y={}, heading={:.1f}".format(x, y, heading))
    return send_command(sock, build_possync(x, y, heading_tenths))


def sync_robot_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)
    if pose is None:
        print("Could not detect robot pose from camera")
        return False
    return send_pose_sync(sock, pose)


def turn_robot_from_camera_pose(sock, pose, turn_angle):
    if pose is None:
        return False
    if abs(turn_angle) < 1.0:
        return True
    if not send_pose_sync(sock, pose):
        return False
    print("Sending camera-corrected TURN {:.1f}".format(turn_angle))
    return send_command(sock, build_turn(int(round(turn_angle)), FINAL_TURN_SPEED))


def goto_center_target(sock, pose, target_row_col):
    if pose is None or target_row_col is None:
        return False

    center_x, center_y = pose_center_xy(pose)
    target_x = int(round(target_row_col[1]))
    target_y = int(round(target_row_col[0]))
    drive_distance = math.hypot(target_x - center_x, target_y - center_y)

    if drive_distance < PICKUP_MIN_FORWARD_STEP:
        print("Skipping tiny GOTO distance {:.1f}".format(drive_distance))
        return True

    if not send_pose_sync(sock, pose):
        return False

    print("Sending CENTER GOTO x={}, y={} (distance {:.1f})".format(target_x, target_y, drive_distance))
    return send_command(sock, build_goto(target_x, target_y))


def align_robot_to_ball(sock, camera, ball_color, target_hint=None):
    """
    Repeatedly turn and re-check the camera until the robot is actually facing
    the ball. This fixes open-loop under-turning before the pickup drive.
    """
    latest_ball = target_hint

    for attempt in range(PICKUP_MAX_TURN_ATTEMPTS):
        state, ball_point = get_pickup_state(camera, ball_color, latest_ball)
        if state is None or ball_point is None:
            return False, latest_ball

        pose = state["pose"]
        latest_ball = ball_point
        desired_heading = heading_from_pose_to_ball(pose, ball_point)
        turn_angle = normalize_turn(desired_heading - pose[2])

        print(
            "Pickup alignment {}: current={:.1f}, desired={:.1f}, error={:.1f}".format(
                attempt + 1,
                pose[2],
                desired_heading,
                turn_angle,
            )
        )

        if abs(turn_angle) <= PICKUP_HEADING_TOLERANCE:
            return True, ball_point

        if not turn_robot_from_camera_pose(sock, pose, turn_angle):
            return False, ball_point

        time.sleep(SYNC_DELAY_SECONDS)

    print("Pickup alignment failed to reach tolerance")
    return False, latest_ball


def approach_until_ball_is_in_claw_range(sock, camera, ball_color, target_hint=None):
    """
    Approach in short camera-corrected steps. Never close from far away and never
    send a blind final forward push.
    """
    latest_ball = target_hint

    for step_index in range(PICKUP_MAX_APPROACH_STEPS):
        state, ball_point = get_pickup_state(camera, ball_color, latest_ball)
        if state is None:
            return False, latest_ball

        pose = state["pose"]
        grappler_point = state["grappler"]

        if pose is None:
            return False, latest_ball

        if ball_point is None:
            print("Ball not visible during pickup approach; assuming it is in the claw mouth")
            return True, latest_ball

        latest_ball = ball_point

        if grappler_point is not None:
            grappler_distance = point_distance(grappler_point, ball_point)
            print("Pickup step {}: grappler-to-ball distance {:.1f}".format(step_index + 1, grappler_distance))
            if grappler_distance <= PICKUP_READY_TO_CLOSE_DISTANCE:
                print("Ball is close enough to close the claw")
                return True, ball_point
        else:
            grappler_distance = None
            print("Pickup step {}: grappler not visible".format(step_index + 1))

        desired_heading = heading_from_pose_to_ball(pose, ball_point)
        turn_angle = normalize_turn(desired_heading - pose[2])

        if abs(turn_angle) > PICKUP_HEADING_TOLERANCE:
            print("Heading is not aligned before approach; correcting before driving")
            ok, latest_ball = align_robot_to_ball(sock, camera, ball_color, ball_point)
            if not ok:
                return False, latest_ball
            continue

        center_x, center_y = pose_center_xy(pose)
        ball_x, ball_y = row_col_to_xy(ball_point)
        center_to_ball = math.hypot(ball_x - center_x, ball_y - center_y)

        if center_to_ball < 1.0:
            print("Pickup cancelled: robot center is on top of ball estimate")
            return False, ball_point

        grappler_offset = estimate_grappler_offset_distance(pose, grappler_point)
        desired_center_to_ball = grappler_offset + PICKUP_GRAPPLER_STOP_SHORT
        forward_distance = center_to_ball - desired_center_to_ball

        print(
            "Pickup approach geometry: center_to_ball={:.1f}, desired_center_to_ball={:.1f}, "
            "forward_distance={:.1f}".format(
                center_to_ball,
                desired_center_to_ball,
                forward_distance,
            )
        )

        if forward_distance <= PICKUP_MIN_FORWARD_STEP:
            if grappler_distance is None:
                print("Geometry says the ball should be in the claw mouth; grappler hidden, so closing")
                return True, ball_point
            if grappler_distance <= PICKUP_READY_TO_CLOSE_DISTANCE:
                print("Close enough after camera check; closing without another drive")
                return True, ball_point
            print(
                "Pickup cancelled: ball is still {:.1f} px from the grappler, so not closing".format(
                    grappler_distance
                )
            )
            return False, ball_point

        drive_step = min(forward_distance, PICKUP_MAX_FORWARD_STEP)
        unit_x = (ball_x - center_x) / center_to_ball
        unit_y = (ball_y - center_y) / center_to_ball
        target_x = center_x + drive_step * unit_x
        target_y = center_y + drive_step * unit_y
        target_x, target_y = clamp_xy_to_map(target_x, target_y)
        target_row_col = xy_to_row_col(target_x, target_y)

        print(
            "Pickup step {}: driving only {:.1f} px toward ball to center target ({}, {})".format(
                step_index + 1,
                drive_step,
                target_row_col[1],
                target_row_col[0],
            )
        )

        if not goto_center_target(sock, pose, target_row_col):
            return False, ball_point

        time.sleep(SYNC_DELAY_SECONDS)

    print("Pickup cancelled: maximum camera-corrected approach steps reached")
    return False, latest_ball


def confirm_ball_target(camera, ball_color, target_hint=None):
    confirmed_target = target_hint

    for _ in range(BALL_CONFIRM_FRAMES):
        state = capture_color_state(camera, PICKUP_RECHECK_IMAGE_PATH)
        if state is None:
            return None

        reference_point = ball_reference_from_state(state)
        candidate = choose_pickup_ball(state["matrix"], ball_color, reference_point, confirmed_target)
        if candidate is None:
            return None

        if confirmed_target is not None:
            movement = point_distance(confirmed_target, candidate)
            if movement > BALL_CONFIRM_MAX_MOVE:
                print(
                    "Ball confirmation failed: target moved {:.1f} px from {} to {}".format(
                        movement,
                        confirmed_target,
                        candidate,
                    )
                )
                return None

        confirmed_target = candidate
        time.sleep(BALL_CONFIRM_DELAY_SECONDS)

    print("Confirmed pickup target:", confirmed_target)
    return confirmed_target


def approach_ball_and_close_claw(sock, camera, ball_color, target_hint=None):
    confirmed_target = confirm_ball_target(camera, ball_color, target_hint)
    if confirmed_target is None:
        print("Pickup cancelled: ball was not confirmed in fresh camera frames")
        return False

    print("Opening claw before pickup approach")
    if not send_command(sock, build_claw_open()):
        return False
    time.sleep(CLAW_OPEN_WAIT_SECONDS)

    print("Aligning robot to ball before driving")
    ok, updated_target = align_robot_to_ball(sock, camera, ball_color, confirmed_target)
    if not ok:
        return False

    print("Approaching ball with camera feedback")
    ready_to_close, updated_target = approach_until_ball_is_in_claw_range(
        sock,
        camera,
        ball_color,
        target_hint=updated_target,
    )
    if not ready_to_close:
        print("Pickup cancelled before closing claw")
        return False

    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False
    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing claw from stopped camera-corrected pose")
    return send_command(sock, build_claw_close())


def goto_then_sync(sock, camera, row, col):
    x = int(round(col))
    y = int(round(row))
    print("Sending GOTO x={}, y={}".format(x, y))
    if not send_command(sock, build_goto(x, y)):
        return False
    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


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


def run_autonomous_camera():
    allocated_time = 1
    start_delay = 2
    begin_time = time.time()
    last_picture_time = time.time()

    camera = open_camera(CAMERA_INDEX)
    if camera is None:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((HOST, PORT))
        send_command(sock, build_handshake())

        count = 0
        ball_collected = False

        while camera.isOpened():
            res, frame = camera.read()
            if not res:
                continue

            warped_frame = warp_frame(frame)

            if time.time() - begin_time >= start_delay:
                if time.time() - last_picture_time >= allocated_time:
                    last_picture_time = time.time()

                    image_name = "{}.png".format(count)
                    full_path = os.path.join(IMAGE_DIR, image_name)
                    cv.imwrite(full_path, warped_frame)
                    count += 1
                    print("Vi tager et billede")

                    color_matrix = create_matrix(full_path)
                    white_list = ball_pos_approx_shape(color_matrix, "W")
                    grappler_point = grapler_pos_approx(color_matrix, "G")
                    print("grappler_point", grappler_point)

                    if grappler_point is None:
                        print("No grappler detected; using robot center as ball-selection reference")
                        pose = robot_pose_approx(color_matrix)
                        if pose is not None:
                            center_x, center_y = pose_center_xy(pose)
                            grappler_point = xy_to_row_col(center_x, center_y)

                    if grappler_point is None:
                        print("No robot reference detected; cannot collect ball")
                        continue

                    if not white_list:
                        print("No white balls detected")
                        continue

                    distance_list = [point_distance(grappler_point, item) for item in white_list]
                    print("ball distances", [round(value, 1) for value in distance_list])

                    paired = list(zip(distance_list, white_list))
                    paired.sort()
                    white_list = [item for _, item in paired]
                    print("white_list", white_list)

                    if not ball_collected:
                        ball_collected = approach_ball_and_close_claw(
                            sock,
                            camera,
                            "W",
                            target_hint=white_list[0],
                        )

                    robot_pixels = robot_pos(color_matrix)
                    robot_pose = robot_pose_approx(color_matrix)
                    print("robot_pixels_count", len(robot_pixels))
                    print("robot_pose", robot_pose)

                    goals = goals_pos_approx(color_matrix, "PK", "C")
                    if goals is not None:
                        goal_a, goal_b = goals
                        print("Goal_A:", goal_a)
                        print("Goal_B:", goal_b)
                    else:
                        print("Goals not detected")

                    orangeball_pos = ball_pos_approx_shape(color_matrix, "O")
                    print("orangeball_pos:", orangeball_pos)

            cv.imshow("camera", warped_frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        sock.close()
        close_camera(camera)


if __name__ == "__main__":
    run_autonomous_camera()