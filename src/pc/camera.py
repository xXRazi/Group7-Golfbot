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
    build_turn,
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

# Stop this many warped-image/map units before the ball center, measured from
# the green grappler marker / claw reference point, not from the robot center.
# Start with 18-22 and tune on the real robot. Larger values stop earlier.
PICKUP_STOP_DISTANCE = 10

# The green grappler marker is in front of the robot center. A* was previously
# planned for this marker, but EV3 GOTO moves the robot center. These fallback
# values are only used if the offset cannot be measured from the current image.
GRAPPLER_FORWARD_OFFSET_FALLBACK = 40.0
GRAPPLER_LATERAL_OFFSET_FALLBACK = 0.0

# Use smaller waypoints during pickup so camera feedback can correct drift before
# the claw reaches the ball.
PICKUP_WAYPOINT_STEP_SIZE = 40

# Extra delay after forced stop, before closing the claw.
PICKUP_SETTLE_SECONDS = 0.15

# First drive only to a safe pre-approach point. The last part of pickup is
# handled by a fresh camera servo loop, because the ball/grappler geometry can
# change a lot after the robot turns.
PICKUP_PREAPPROACH_DISTANCE = 55.0

# Final camera-servo tuning.
#
# Important: PICKUP_STOP_DISTANCE above is used for the coarse A* approach, not
# for the last few centimeters of pickup. In v6 the final close distance was
# too large, so the robot could stop with the ball still just outside the claw.
# v7 drives closer, then performs one small forward "scoop" before closing.
PICKUP_SERVO_MAX_ITERATIONS = 18
PICKUP_SERVO_MAX_FORWARD_STEP = 14.0
PICKUP_SERVO_MIN_FORWARD_STEP = 3.0

# Close when the robot center is roughly this far from the ball. If the robot
# still stops with the ball outside the fingers, lower this by 2-3. If it pushes
# the ball away before closing, raise it by 2-3.
PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE = 36.0
PICKUP_CENTER_TO_BALL_MARGIN = 3.0
PICKUP_FINAL_HEADING_CLOSE_TOLERANCE = 14.0
PICKUP_DIRECT_FALLBACK_MAX_DISTANCE = 90.0

# After the camera says the robot is close, move this many extra map pixels
# forward before closing. This intentionally puts the ball slightly inside the
# fingers instead of right at the outside edge. Tune this first.
PICKUP_FINAL_SCOOP_DISTANCE = 6.0

# After the robot center reaches the pickup pose, rotate in place so the claw
# actually points at the ball. GOTO only controls position; without this final
# turn the robot can stop beside the ball and close the claw while facing the
# wrong way.
PICKUP_FINAL_HEADING_TOLERANCE = 5.0
PICKUP_FINAL_SYNC_DELAY_SECONDS = 0.15

# Final camera-based nudge: after aligning, if the grappler is still much too
# far from the ball, drive a short distance forward along the current heading.
# Keep this small so the robot cannot ram the ball/wall if detection is noisy.
PICKUP_FINAL_NUDGE_MAX_DISTANCE = 28.0
PICKUP_FINAL_NUDGE_MARGIN = 5.0


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


def sync_robot_pose_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return None

    x, y, heading = pose
    x = int(round(x))
    y = int(round(y))
    heading_tenths = int(round(heading * 10))

    print("Camera sync: x={}, y={}, heading={:.1f}".format(x, y, heading))

    if not send_command(sock, build_possync(x, y, heading_tenths)):
        return None

    return x, y, heading


def sync_robot_from_camera(sock, camera):
    return sync_robot_pose_from_camera(sock, camera) is not None


def sync_robot_pose_value(sock, robot_pose, label="Camera pose"):
    """Send an already-detected camera pose to the EV3 odometry state."""
    if robot_pose is None:
        return False

    x, y, heading = robot_pose
    x = int(round(x))
    y = int(round(y))
    heading_tenths = int(round(float(heading) * 10))

    print("{} sync: x={}, y={}, heading={:.1f}".format(label, x, y, heading))
    return send_command(sock, build_possync(x, y, heading_tenths))


def normalize_turn_angle(angle):
    """Normalize a heading correction to the shortest signed turn."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def turn_robot_to_heading(sock, camera, target_heading, tolerance_degrees=PICKUP_FINAL_HEADING_TOLERANCE):
    """Use camera sync, then rotate the robot to a requested map/image heading."""
    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return False

    _x, _y, current_heading = synced_pose
    turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))

    print(
        "Pickup final alignment: current_heading={:.1f}, target_heading={:.1f}, "
        "turn_angle={:.1f}".format(
            current_heading,
            target_heading,
            turn_angle,
        )
    )

    if abs(turn_angle) <= float(tolerance_degrees):
        print("Pickup final alignment: already within tolerance")
        return True

    if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
        return False

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


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


def map_point_is_valid(point):
    row, col = point
    return 0 <= row < height and 0 <= col < width


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_map_point(point, margin=2):
    row, col = point
    return (
        int(round(clamp(row, margin, height - 1 - margin))),
        int(round(clamp(col, margin, width - 1 - margin))),
    )


def heading_from_map_points(start, end):
    """Return image/map heading from start(row, col) to end(row, col)."""
    start_row, start_col = start
    end_row, end_col = end
    delta_row = float(end_row - start_row)
    delta_col = float(end_col - start_col)

    if delta_row == 0.0 and delta_col == 0.0:
        return 0.0

    return math.degrees(math.atan2(delta_row, delta_col)) % 360.0


def estimate_grappler_offset(robot_pose, grappler_point):
    """
    Estimate where the green grappler marker is relative to the robot center.

    A* detects/plans from the green marker, but EV3 GOTO controls the robot
    center. Without compensating for this offset, the robot center is driven to
    the claw waypoint and the actual claw overshoots/misses the ball.
    """
    fallback = (
        GRAPPLER_FORWARD_OFFSET_FALLBACK,
        GRAPPLER_LATERAL_OFFSET_FALLBACK,
    )

    if robot_pose is None or grappler_point is None:
        print(
            "Grappler offset: using fallback forward={:.1f}, lateral={:.1f}".format(
                fallback[0],
                fallback[1],
            )
        )
        return fallback

    center_x, center_y, heading = robot_pose
    grappler_row, grappler_col = grappler_point

    dx = float(grappler_col) - float(center_x)
    dy = float(grappler_row) - float(center_y)

    heading_rad = math.radians(heading)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    # Forward is along the robot heading. Lateral is positive to the robot's left.
    forward = dx * cos_h + dy * sin_h
    lateral = dx * (-sin_h) + dy * cos_h

    if not (20.0 <= forward <= 80.0) or abs(lateral) > 30.0:
        print(
            "Grappler offset looked wrong: forward={:.1f}, lateral={:.1f}; "
            "using fallback forward={:.1f}, lateral={:.1f}".format(
                forward,
                lateral,
                fallback[0],
                fallback[1],
            )
        )
        return fallback

    print(
        "Grappler offset: forward={:.1f}, lateral={:.1f}".format(
            forward,
            lateral,
        )
    )
    return forward, lateral


def center_point_for_grappler_point(grappler_point, heading, forward_offset, lateral_offset):
    """Convert desired grappler marker point into desired robot-center point."""
    grappler_row, grappler_col = grappler_point
    heading_rad = math.radians(heading)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    center_col = (
        float(grappler_col)
        - forward_offset * cos_h
        + lateral_offset * sin_h
    )
    center_row = (
        float(grappler_row)
        - forward_offset * sin_h
        - lateral_offset * cos_h
    )

    return int(round(center_row)), int(round(center_col))


def robot_center_point(robot_pose):
    center_x, center_y, _heading = robot_pose
    return int(round(center_y)), int(round(center_x))


def build_center_pickup_plan(
    color_matrix,
    grappler_to_ball_path,
    current_robot_pose,
    current_grappler_point,
):
    """
    Build a coarse pre-approach plan for the robot center.

    The previous version tried to compute the exact final pickup pose up front.
    That is fragile: after the robot turns, the green grappler marker moves
    around the robot center, so the final claw/ball geometry can be very
    different from the first image. This function now only moves the robot to a
    safe point near the ball. A camera-servo loop does the final turn and
    forward approach from fresh images.
    """
    if current_robot_pose is None or current_grappler_point is None:
        print("Cannot build pickup pre-approach without robot pose and grappler")
        return None

    preapproach_grappler_path = truncate_path_before_target(
        grappler_to_ball_path,
        PICKUP_PREAPPROACH_DISTANCE,
    )

    if not path_is_valid(preapproach_grappler_path):
        return None

    ball_point = grappler_to_ball_path[-1]
    grappler_preapproach_point = preapproach_grappler_path[-1]
    center_start_point = robot_center_point(current_robot_pose)

    # Translate the robot center by the same map delta that we want the green
    # grappler marker to move. Do not use the final desired heading here; that
    # was what made the target jump to surprising places when the robot still
    # needed to rotate.
    delta_row = grappler_preapproach_point[0] - current_grappler_point[0]
    delta_col = grappler_preapproach_point[1] - current_grappler_point[1]
    center_stop_point = (
        int(round(center_start_point[0] + delta_row)),
        int(round(center_start_point[1] + delta_col)),
    )

    final_heading = heading_from_map_points(grappler_preapproach_point, ball_point)
    forward_offset, lateral_offset = estimate_grappler_offset(
        current_robot_pose,
        current_grappler_point,
    )

    print(
        "Pickup pre-approach plan: ball={}, grappler_now={}, grappler_pre={}, "
        "center_start={}, center_stop={}, estimated_final_heading={:.1f}".format(
            ball_point,
            current_grappler_point,
            grappler_preapproach_point,
            center_start_point,
            center_stop_point,
            final_heading,
        )
    )

    if not map_point_is_valid(center_stop_point):
        clamped_stop_point = clamp_map_point(center_stop_point, margin=5)
        print(
            "Center pickup pre-approach target was out of bounds: {}; "
            "using clamped target {}".format(center_stop_point, clamped_stop_point)
        )
        center_stop_point = clamped_stop_point

    center_path = A_star(color_matrix, center_start_point, center_stop_point)

    if not path_is_valid(center_path):
        direct_distance = point_distance(center_start_point, center_stop_point)
        print(
            "Center pickup A* failed: {}. Using direct coarse move; distance={:.1f}".format(
                center_path,
                direct_distance,
            )
        )
        center_path = [center_start_point, center_stop_point]

    return {
        "center_path": center_path,
        "final_heading": final_heading,
        "ball_point": ball_point,
        "grappler_stop_point": grappler_preapproach_point,
        "center_stop_point": center_stop_point,
        "forward_offset": forward_offset,
        "lateral_offset": lateral_offset,
    }

def build_center_pickup_path(
    color_matrix,
    grappler_to_ball_path,
    current_robot_pose,
    current_grappler_point,
):
    """Compatibility wrapper returning only the center path."""
    plan = build_center_pickup_plan(
        color_matrix,
        grappler_to_ball_path,
        current_robot_pose,
        current_grappler_point,
    )

    if plan is None:
        return None

    return plan["center_path"]

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


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def simplify_path_for_robot(robot_path, min_spacing=40):
    """Reduce one-pixel A* paths to a few stable GOTO waypoints."""
    if not path_is_valid(robot_path):
        return robot_path

    if len(robot_path) <= 2:
        return robot_path

    min_spacing = max(1.0, float(min_spacing))
    waypoints = [robot_path[0]]
    previous_direction = None

    for index in range(1, len(robot_path)):
        previous_point = robot_path[index - 1]
        current_point = robot_path[index]
        current_direction = (
            sign(current_point[0] - previous_point[0]),
            sign(current_point[1] - previous_point[1]),
        )

        direction_changed = (
            previous_direction is not None
            and current_direction != previous_direction
        )
        far_enough = point_distance(waypoints[-1], current_point) >= min_spacing

        if direction_changed and point_distance(waypoints[-1], previous_point) >= 8.0:
            if waypoints[-1] != previous_point:
                waypoints.append(previous_point)
        elif far_enough:
            waypoints.append(current_point)

        previous_direction = current_direction

    if waypoints[-1] != robot_path[-1]:
        waypoints.append(robot_path[-1])

    # Remove accidental tiny final hops that make the robot twitch.
    cleaned = [waypoints[0]]
    for point in waypoints[1:]:
        if point == robot_path[-1] or point_distance(cleaned[-1], point) >= 6.0:
            cleaned.append(point)

    print(
        "Simplified path: raw_points={}, waypoints={}".format(
            len(robot_path),
            len(cleaned),
        )
    )
    return cleaned


def follow_path_with_camera_sync(sock, camera, robot_path, step_size=10):
    if not path_is_valid(robot_path):
        print("Invalid path:", robot_path)
        return False

    if not sync_robot_from_camera(sock, camera):
        return False

    waypoints = simplify_path_for_robot(robot_path, min_spacing=step_size)

    # The first waypoint is usually the current position. Sending a GOTO to it
    # only makes the EV3 do a tiny corrective turn, which feels like jitter.
    if len(waypoints) > 1:
        waypoints = waypoints[1:]

    for row, col in waypoints:
        if not goto_then_sync(sock, camera, row, col):
            return False

    return True


def choose_closest_ball_to_grappler(balls, grappler_point):
    if not balls or grappler_point is None:
        return None

    return min(balls, key=lambda ball: point_distance(ball, grappler_point))


def capture_pickup_scene(camera, ball_color="W"):
    """Read one warped frame and return detection results used for pickup."""
    warped_frame = read_warped_frame(camera)

    if warped_frame is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        cv.imwrite(temp_path, warped_frame)
        color_matrix = create_matrix(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    robot_pose = robot_pose_approx(color_matrix)
    grappler_point = grapler_pos_approx(color_matrix, "G")
    balls = ball_pos_approx_shape(color_matrix, ball_color)
    ball_point = choose_closest_ball_to_grappler(balls, grappler_point)

    return {
        "color_matrix": color_matrix,
        "robot_pose": robot_pose,
        "grappler_point": grappler_point,
        "balls": balls,
        "ball_point": ball_point,
    }


def drive_toward_map_point(sock, camera, robot_pose, target_point, distance_map_units):
    """
    Move the robot center a short distance toward target_point.

    This deliberately sends one GOTO per camera frame. GOTO turns toward the
    target and then drives, so we do not need a separate TURN command for every
    tiny heading error.
    """
    center_x, center_y, _heading = robot_pose
    center_point = (int(round(center_y)), int(round(center_x)))
    distance_to_target = point_distance(center_point, target_point)

    if distance_to_target <= 0.001:
        print("Pickup servo: center is already at target point")
        return True

    distance_map_units = float(distance_map_units)
    unit_row = (float(target_point[0]) - float(center_point[0])) / distance_to_target
    unit_col = (float(target_point[1]) - float(center_point[1])) / distance_to_target

    target_row = int(round(float(center_point[0]) + distance_map_units * unit_row))
    target_col = int(round(float(center_point[1]) + distance_map_units * unit_col))

    if not map_point_is_valid((target_row, target_col)):
        clamped_row, clamped_col = clamp_map_point((target_row, target_col), margin=5)
        print(
            "Pickup servo center target was out of bounds: {}; "
            "using clamped target ({}, {})".format(
                (target_row, target_col),
                clamped_col,
                clamped_row,
            )
        )
        target_row, target_col = clamped_row, clamped_col

    print(
        "Pickup servo: center->ball distance={:.1f}; moving {:.1f} to center=({}, {})".format(
            distance_to_target,
            distance_map_units,
            target_col,
            target_row,
        )
    )

    # The target was computed from this camera frame. Sync that exact pose to
    # the EV3 before sending GOTO, otherwise the EV3 may use stale odometry from
    # the previous turn/sync and drive to a surprising place.
    if not sync_robot_pose_value(sock, robot_pose, label="Pickup servo pre-GOTO"):
        return False

    if not send_command(sock, build_goto(target_col, target_row)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def final_scoop_forward_before_close(sock, camera, robot_pose, distance_map_units=PICKUP_FINAL_SCOOP_DISTANCE):
    """Move a small distance straight forward to put the ball inside the fingers."""
    if distance_map_units <= 0.0:
        return True

    center_x, center_y, heading = robot_pose
    heading_rad = math.radians(float(heading))
    target_x = int(round(float(center_x) + float(distance_map_units) * math.cos(heading_rad)))
    target_y = int(round(float(center_y) + float(distance_map_units) * math.sin(heading_rad)))

    target_row, target_col = clamp_map_point((target_y, target_x), margin=5)
    target_x = target_col
    target_y = target_row

    print(
        "Pickup final scoop: moving forward {:.1f} map units to center=({}, {}) before closing".format(
            distance_map_units,
            target_x,
            target_y,
        )
    )

    if not sync_robot_pose_value(sock, robot_pose, label="Pickup final scoop pre-GOTO"):
        return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def servo_align_and_approach_ball(sock, camera, ball_color="W"):
    """
    Final pickup controller.

    v5 used the vector from the green grappler marker to the ball as the main
    control signal. In the logs from the failed test that marker jumped around,
    which caused repeated turns and only a few forward moves. This version uses
    the robot center and the ball center for the final approach. The EV3 GOTO
    command already turns the robot toward the center target before driving, so
    this avoids the turn/turn/turn jitter and keeps approaching until the robot
    front should be close enough for the claw to close.
    """
    last_scene = None

    for iteration in range(1, PICKUP_SERVO_MAX_ITERATIONS + 1):
        scene = capture_pickup_scene(camera, ball_color=ball_color)
        last_scene = scene

        if scene is None:
            print("Pickup servo: could not read camera frame")
            return False

        robot_pose = scene["robot_pose"]
        grappler_point = scene["grappler_point"]
        ball_point = scene["ball_point"]

        if robot_pose is None:
            print("Pickup servo: missing robot detection")
            return False

        # If the ball disappears only after we have driven near it, it is often
        # hidden by the claw/robot. Close instead of starting another attempt.
        if ball_point is None:
            print(
                "Pickup servo: ball is no longer visible; assuming it is at/inside the claw and closing"
            )
            return True

        center_x, center_y, current_heading = robot_pose
        center_point = (int(round(center_y)), int(round(center_x)))
        center_to_ball = point_distance(center_point, ball_point)
        target_heading = heading_from_map_points(center_point, ball_point)
        heading_error = normalize_turn_angle(target_heading - current_heading)

        if grappler_point is not None:
            grappler_to_ball = point_distance(grappler_point, ball_point)
            grappler_text = ", grappler={}, grappler_distance={:.1f}".format(
                grappler_point,
                grappler_to_ball,
            )
        else:
            grappler_text = ", grappler=None"

        print(
            "Pickup servo iteration {}: center={}, ball={}, center_distance={:.1f}, "
            "heading={:.1f}, target_heading={:.1f}, heading_error={:.1f}{}".format(
                iteration,
                center_point,
                ball_point,
                center_to_ball,
                current_heading,
                target_heading,
                heading_error,
                grappler_text,
            )
        )

        # The front/claw is roughly this many map pixels in front of the robot
        # center. When the center is this far from the ball and facing it, the
        # ball should be between the fingers. This is easier to tune than the
        # green-marker distance, because the green marker was noisy in testing.
        if center_to_ball <= PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE + PICKUP_CENTER_TO_BALL_MARGIN:
            if abs(heading_error) > PICKUP_FINAL_HEADING_CLOSE_TOLERANCE:
                print("Pickup servo: close to ball, doing one final heading correction")
                if not turn_robot_to_heading(
                    sock,
                    camera,
                    target_heading,
                    tolerance_degrees=6.0,
                ):
                    return False

                # Turning in place moves the claw around the center. Re-read the
                # camera instead of closing immediately from a stale geometry.
                continue

            print("Pickup servo: center is close enough; doing final scoop before claw close")
            return final_scoop_forward_before_close(sock, camera, robot_pose)

        forward_distance = center_to_ball - PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE
        forward_distance = min(forward_distance, PICKUP_SERVO_MAX_FORWARD_STEP)

        if forward_distance < PICKUP_SERVO_MIN_FORWARD_STEP:
            print("Pickup servo: remaining center move is tiny; closing")
            return True

        if not drive_toward_map_point(sock, camera, robot_pose, ball_point, forward_distance):
            return False

    # Final safety check. Do not close if the camera still clearly sees that the
    # robot center is far from the ball. Close only if the remaining error is
    # small enough that the claw should be around the ball.
    final_scene = capture_pickup_scene(camera, ball_color=ball_color)
    if final_scene is not None:
        final_robot = final_scene["robot_pose"]
        final_ball = final_scene["ball_point"]
        if final_robot is not None and final_ball is not None:
            final_center = (int(round(final_robot[1])), int(round(final_robot[0])))
            final_distance = point_distance(final_center, final_ball)
            print("Pickup servo: max iterations reached; final center distance={:.1f}".format(final_distance))
            if final_distance > PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE + 8.0:
                print("Pickup servo: still too far from ball; not closing claw")
                return False

            if final_robot is not None:
                return final_scoop_forward_before_close(sock, camera, final_robot)

    print("Pickup servo: max iterations reached but final distance is acceptable; closing")
    return True

def final_pickup_camera_nudge(sock, camera, ball_color="W", allow_extra_turn=True):
    """
    Make one small camera-based forward correction after final alignment.

    This is deliberately conservative. It only moves forward if the ball is in
    front of the grappler and the measured distance says the claw is still too
    far away.
    """
    warped_frame = read_warped_frame(camera)

    if warped_frame is None:
        return False

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        cv.imwrite(temp_path, warped_frame)
        color_matrix = create_matrix(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    robot_pose = robot_pose_approx(color_matrix)
    grappler_point = grapler_pos_approx(color_matrix, "G")
    balls = ball_pos_approx_shape(color_matrix, ball_color)
    ball_point = choose_closest_ball_to_grappler(balls, grappler_point)

    if robot_pose is None or grappler_point is None or ball_point is None:
        print("Pickup final nudge: missing robot, grappler, or ball detection")
        return True

    ball_heading = heading_from_map_points(grappler_point, ball_point)
    _center_x, _center_y, current_heading = robot_pose
    heading_error = normalize_turn_angle(ball_heading - current_heading)

    print(
        "Pickup final nudge: grappler={}, ball={}, distance={:.1f}, "
        "ball_heading={:.1f}, robot_heading={:.1f}, heading_error={:.1f}".format(
            grappler_point,
            ball_point,
            point_distance(grappler_point, ball_point),
            ball_heading,
            current_heading,
            heading_error,
        )
    )

    # If the ball is not mostly in front of the claw, rotate once more instead
    # of driving sideways into it. Then re-run this check from a fresh image so
    # the distance is measured after the rotation.
    if abs(heading_error) > 12.0:
        if not allow_extra_turn:
            print("Pickup final nudge: heading still off after extra turn; not nudging")
            return True

        if not turn_robot_to_heading(sock, camera, ball_heading, tolerance_degrees=4.0):
            return False

        return final_pickup_camera_nudge(sock, camera, ball_color, allow_extra_turn=False)

    center_x, center_y, current_heading = robot_pose

    distance_to_ball = point_distance(grappler_point, ball_point)
    desired_distance = float(PICKUP_STOP_DISTANCE)
    forward_distance = distance_to_ball - desired_distance

    if forward_distance <= PICKUP_FINAL_NUDGE_MARGIN:
        print("Pickup final nudge: distance is already good")
        return True

    forward_distance = min(forward_distance, float(PICKUP_FINAL_NUDGE_MAX_DISTANCE))
    heading_rad = math.radians(current_heading)
    target_x = int(round(float(center_x) + forward_distance * math.cos(heading_rad)))
    target_y = int(round(float(center_y) + forward_distance * math.sin(heading_rad)))

    if not (0 <= target_x < width and 0 <= target_y < height):
        print("Pickup final nudge target is out of bounds:", (target_x, target_y))
        return True

    print(
        "Pickup final nudge: driving forward {:.1f} map units to center=({}, {})".format(
            forward_distance,
            target_x,
            target_y,
        )
    )

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)

def approach_ball_and_close_claw(
    sock,
    camera,
    color_matrix,
    grappler_to_ball_path,
    current_grappler_point=None,
    current_robot_pose=None,
):
    """
    Open the claw, drive to a stable pre-approach point, then use fresh camera
    feedback for the final alignment and forward pickup motion.
    """
    pickup_plan = build_center_pickup_plan(
        color_matrix,
        grappler_to_ball_path,
        current_robot_pose,
        current_grappler_point,
    )

    center_pickup_path = None

    if pickup_plan is not None and path_is_valid(pickup_plan["center_path"]):
        center_pickup_path = pickup_plan["center_path"]
    else:
        print(
            "Pickup pre-approach plan failed; opening claw and using "
            "camera-servo pickup from the current position"
        )

    print("Opening claw before pickup approach")
    if not send_command(sock, build_claw_open()):
        return False

    if center_pickup_path is not None:
        if not follow_path_with_camera_sync(
            sock,
            camera,
            center_pickup_path,
            step_size=PICKUP_WAYPOINT_STEP_SIZE,
        ):
            return False

    # Do not trust the first image for the final pickup pose. The robot may have
    # slipped or rotated during the coarse move. Use a fresh stop-and-check servo
    # loop so the claw actually points at the ball before closing.
    if not servo_align_and_approach_ball(sock, camera, ball_color="W"):
        print("Pickup servo failed; not closing claw blindly")
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

                    current_robot_pose = robot_pose_approx(color_matrix)

                    if current_robot_pose is None:
                        print("No robot pose detected; cannot collect ball")
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
                        # Mark the pickup sequence as executed as soon as the
                        # attempt starts. Previously this flag only became True
                        # if every final camera check succeeded. If a final
                        # detection missed the ball, the outer camera loop would
                        # immediately call approach_ball_and_close_claw() again,
                        # which sends another CLAW_OPEN command. That is the
                        # "it opened again instead of closing" behaviour.
                        path_executed = True
                        pickup_success = approach_ball_and_close_claw(
                            sock,
                            camera,
                            color_matrix,
                            robot_path,
                            current_grappler_point=grapler_point,
                            current_robot_pose=current_robot_pose,
                        )

                        if not pickup_success:
                            print(
                                "Pickup attempt did not finish cleanly; not retrying automatically because that would re-open the claw"
                            )

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