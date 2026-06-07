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
from collection_algorithm import get_h_list
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


# ==========================================
# 1. PERSPECTIVE WARP SETUP
# ==========================================

# Final warped camera/map size: x = 0..639, y = 0..359.
width, height = 640, 360

# Raw camera corners, ordered as:
# top-left, top-right, bottom-right, bottom-left.
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


# ==========================================
# 2. SETTINGS
# ==========================================

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

CAMERA_INDEX = 0
SYNC_DELAY_SECONDS = 0.25
SYNC_IMAGE_PATH = os.path.join(IMAGE_DIR, "robot_sync_frame.png")
PICKUP_RECHECK_IMAGE_PATH = os.path.join(IMAGE_DIR, "pickup_recheck_frame.png")

# The key precision fix:
# EV3 GOTO moves the robot CENTER. It does not move the green grappler point.
# So for pickup we calculate where the robot center must stop so that the
# grappler/front point is on the ball.
DEFAULT_GRAPPLER_OFFSET_DISTANCE = 42.0
MIN_GRAPPLER_OFFSET_DISTANCE = 20.0
MAX_GRAPPLER_OFFSET_DISTANCE = 80.0

# Pull the center a little closer than the exact edge position so the ball is
# slightly inside the claw before closing.
PICKUP_CAPTURE_OVERLAP = 5.0

# When far away, first go to a point behind the pickup pose, re-read the camera,
# then do the final approach.
PICKUP_PRE_APPROACH_DISTANCE = 75.0
PICKUP_PRE_APPROACH_SKIP_DISTANCE = 25.0

# Small movement and correction thresholds, in warped-map pixels.
PICKUP_MIN_GOTO_DISTANCE = 4.0
PICKUP_GRAPPLER_TOLERANCE = 7.0
PICKUP_TARGET_MARGIN = 8.0

# EV3 claw commands are non-blocking, so give the medium motor time to move.
CLAW_OPEN_WAIT_SECONDS = 0.45
PICKUP_SETTLE_SECONDS = 0.20
FINAL_TURN_SPEED = 15


# ==========================================
# 3. CAMERA HELPERS
# ==========================================

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
    """Capture a warped frame, save it, and return matrix + robot pose info."""
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

    if warped_frame is None:
        return

    cv.imshow("camera", warped_frame)
    cv.waitKey(1)


# ==========================================
# 4. GEOMETRY HELPERS
# ==========================================

def path_is_valid(robot_path):
    return robot_path and not isinstance(robot_path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def row_col_to_xy(point):
    """Convert a matrix point (row, col) to EV3/map coordinates (x, y)."""
    return float(point[1]), float(point[0])


def xy_to_row_col(x, y):
    """Convert EV3/map coordinates (x, y) to a matrix point (row, col)."""
    return int(round(y)), int(round(x))


def pose_center_xy(pose):
    """robot_pose_approx() returns (center_col, center_row, heading_degrees)."""
    return float(pose[0]), float(pose[1])


def normalize_heading(angle):
    return float(angle) % 360.0


def normalize_turn(angle):
    return (float(angle) + 180.0) % 360.0 - 180.0


def heading_to_unit(heading_degrees):
    radians = math.radians(heading_degrees)
    return math.cos(radians), math.sin(radians)


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
        chosen = closest_point(target_hint, balls)
    else:
        chosen = closest_point(reference_point, balls)

    print("Chosen {} ball: {} from candidates {}".format(ball_color, chosen, balls))
    return chosen


def estimate_grappler_offset_distance(pose, grappler_point):
    """Estimate center-to-grappler distance from the current camera frame."""
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


def compute_center_target_for_ball(pose, ball_point, grappler_point=None, extra_standoff=0.0):
    """
    Return the robot-center target that places the grappler on the ball.

    ball_point is (row, col). The returned target is also (row, col), but it is
    the target for the robot CENTER, because build_goto(x, y) drives the center.
    """
    if pose is None or ball_point is None:
        return None

    center_x, center_y = pose_center_xy(pose)
    ball_x, ball_y = row_col_to_xy(ball_point)

    to_ball_x = ball_x - center_x
    to_ball_y = ball_y - center_y
    distance_to_ball = math.hypot(to_ball_x, to_ball_y)

    if distance_to_ball < 1.0:
        unit_x, unit_y = heading_to_unit(pose[2])
    else:
        unit_x = to_ball_x / distance_to_ball
        unit_y = to_ball_y / distance_to_ball

    offset_distance = estimate_grappler_offset_distance(pose, grappler_point)
    effective_offset = max(0.0, offset_distance - PICKUP_CAPTURE_OVERLAP)
    desired_ball_to_center = effective_offset + float(extra_standoff)

    if extra_standoff > 0.0:
        # Do not command a staging move behind the robot when already close.
        if distance_to_ball <= desired_ball_to_center + PICKUP_PRE_APPROACH_SKIP_DISTANCE:
            print(
                "Already close enough for staging: distance_to_ball={:.1f}, "
                "wanted_staging_distance={:.1f}".format(
                    distance_to_ball,
                    desired_ball_to_center,
                )
            )
            desired_ball_to_center = distance_to_ball
    else:
        # Keep the final pickup movement forward and small instead of overshooting.
        desired_ball_to_center = min(
            desired_ball_to_center,
            max(0.0, distance_to_ball - PICKUP_MIN_GOTO_DISTANCE),
        )

    target_x = ball_x - desired_ball_to_center * unit_x
    target_y = ball_y - desired_ball_to_center * unit_y
    target_x, target_y = clamp_xy_to_map(target_x, target_y)

    target_row_col = xy_to_row_col(target_x, target_y)
    target_heading = normalize_heading(math.degrees(math.atan2(unit_y, unit_x)))

    print(
        "Pickup geometry: center=({:.1f}, {:.1f}), ball=({:.1f}, {:.1f}), "
        "distance_to_ball={:.1f}, desired_ball_to_center={:.1f}, "
        "target_center=({}, {}), target_heading={:.1f}".format(
            center_x,
            center_y,
            ball_x,
            ball_y,
            distance_to_ball,
            desired_ball_to_center,
            target_row_col[1],
            target_row_col[0],
            target_heading,
        )
    )

    return target_row_col, target_heading


# ==========================================
# 5. EV3 COMMAND HELPERS
# ==========================================

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


def goto_center_target(sock, pose, target_row_col):
    """Sync EV3 to camera pose, then drive robot center to target_row_col."""
    if pose is None or target_row_col is None:
        return False

    center_x, center_y = pose_center_xy(pose)
    target_x = int(round(target_row_col[1]))
    target_y = int(round(target_row_col[0]))
    drive_distance = math.hypot(target_x - center_x, target_y - center_y)

    if not send_pose_sync(sock, pose):
        return False

    if drive_distance < PICKUP_MIN_GOTO_DISTANCE:
        print("Skipping tiny GOTO distance {:.1f}".format(drive_distance))
        return True

    print(
        "Sending CENTER GOTO x={}, y={} (distance {:.1f})".format(
            target_x,
            target_y,
            drive_distance,
        )
    )
    return send_command(sock, build_goto(target_x, target_y))


def turn_to_face_ball(sock, pose, ball_point):
    if pose is None or ball_point is None:
        return False

    center_x, center_y = pose_center_xy(pose)
    ball_x, ball_y = row_col_to_xy(ball_point)
    desired_heading = normalize_heading(math.degrees(math.atan2(ball_y - center_y, ball_x - center_x)))
    turn_angle = normalize_turn(desired_heading - pose[2])

    print(
        "Final heading check: current={:.1f}, desired={:.1f}, turn={:.1f}".format(
            pose[2],
            desired_heading,
            turn_angle,
        )
    )

    if abs(turn_angle) < 3.0:
        return True

    return send_command(sock, build_turn(int(round(turn_angle)), FINAL_TURN_SPEED))


def goto_then_sync(sock, camera, row, col):
    """Compatibility helper used by robot_console.py/debugging."""
    x = int(round(col))
    y = int(round(row))

    print("Sending GOTO x={}, y={}".format(x, y))

    if not send_command(sock, build_goto(x, y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def follow_path_with_camera_sync(sock, camera, robot_path, step_size=10):
    """Compatibility helper for non-pickup path debugging."""
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


# ==========================================
# 6. PRECISE BALL PICKUP
# ==========================================

def precise_pickup_step(sock, camera, ball_color, target_hint=None, extra_standoff=0.0):
    """Re-read camera, choose a ball, compute robot-center target, and drive."""
    state = capture_color_state(camera, PICKUP_RECHECK_IMAGE_PATH)

    if state is None:
        return False, None

    pose = state["pose"]
    color_matrix = state["matrix"]
    grappler_point = state["grappler"]

    if pose is None:
        print("Could not detect robot pose for pickup")
        return False, None

    reference_point = grappler_point
    if reference_point is None:
        center_x, center_y = pose_center_xy(pose)
        reference_point = xy_to_row_col(center_x, center_y)

    ball_point = choose_pickup_ball(color_matrix, ball_color, reference_point, target_hint)

    if ball_point is None:
        return False, None

    target_info = compute_center_target_for_ball(
        pose,
        ball_point,
        grappler_point,
        extra_standoff=extra_standoff,
    )

    if target_info is None:
        return False, ball_point

    target_row_col, _target_heading = target_info

    if not goto_center_target(sock, pose, target_row_col):
        return False, ball_point

    return True, ball_point


def correct_if_grappler_still_misaligned(sock, camera, ball_color, target_hint=None):
    """Make one final camera-based correction before closing the claw."""
    state = capture_color_state(camera, PICKUP_RECHECK_IMAGE_PATH)

    if state is None:
        return False

    pose = state["pose"]
    color_matrix = state["matrix"]
    grappler_point = state["grappler"]

    if pose is None:
        print("Could not detect robot pose for final correction")
        return False

    reference_point = grappler_point
    if reference_point is None:
        center_x, center_y = pose_center_xy(pose)
        reference_point = xy_to_row_col(center_x, center_y)

    ball_point = choose_pickup_ball(color_matrix, ball_color, reference_point, target_hint)

    if ball_point is None:
        # The claw may hide the ball. In that case, close from the current pose
        # instead of failing after a likely-successful approach.
        print("Ball not visible in final correction frame; closing from current pose")
        return turn_to_face_ball(sock, pose, target_hint) if target_hint else True

    if grappler_point is not None:
        visible_error = point_distance(grappler_point, ball_point)
        print("Visible grappler-to-ball error: {:.1f}".format(visible_error))

        if visible_error <= PICKUP_GRAPPLER_TOLERANCE:
            return turn_to_face_ball(sock, pose, ball_point)

    target_info = compute_center_target_for_ball(
        pose,
        ball_point,
        grappler_point,
        extra_standoff=0.0,
    )

    if target_info is None:
        return False

    target_row_col, _target_heading = target_info

    if not goto_center_target(sock, pose, target_row_col):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return True


def approach_ball_and_close_claw(sock, camera, ball_color, target_hint=None):
    """Open the claw, approach precisely, correct once, then close."""
    print("Opening claw before pickup approach")
    if not send_command(sock, build_claw_open()):
        return False

    time.sleep(CLAW_OPEN_WAIT_SECONDS)

    print("Moving to pickup staging pose")
    ok, updated_target = precise_pickup_step(
        sock,
        camera,
        ball_color,
        target_hint=target_hint,
        extra_standoff=PICKUP_PRE_APPROACH_DISTANCE,
    )

    if not ok:
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    print("Making final pickup approach")
    ok, updated_target = precise_pickup_step(
        sock,
        camera,
        ball_color,
        target_hint=updated_target,
        extra_standoff=0.0,
    )

    if not ok:
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    if not correct_if_grappler_still_misaligned(
        sock,
        camera,
        ball_color,
        target_hint=updated_target,
    ):
        return False

    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing claw at corrected pickup pose")
    return send_command(sock, build_claw_close())


# ==========================================
# 7. MAIN AUTONOMOUS LOOP
# ==========================================

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
        path_executed = False

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
                        print("No grapler detected; cannot collect ball")
                        continue

                    if not white_list:
                        print("No white balls detected")
                        continue

                    # Choose the nearest white ball to the grappler. This keeps
                    # the old selection behavior, but pickup itself now uses
                    # robot-center geometry instead of a grappler path as GOTO.
                    min_list = []
                    for item in white_list:
                        value = get_h_list(grappler_point[0], grappler_point[1], item[0], item[1])
                        min_list.append(value)

                    print("minlist", min_list)
                    paired = list(zip(min_list, white_list))
                    paired.sort()
                    white_list = [item for _, item in paired]
                    print("white_list", white_list)

                    if not path_executed:
                        path_executed = approach_ball_and_close_claw(
                            sock,
                            camera,
                            "W",
                            target_hint=white_list[0],
                        )

                    robot_position = robot_pos(color_matrix)
                    print("robot_position", robot_position)

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