import math
import time

from camera import create_matrix_from_frame, detect_vision_from_raw_frame, read_arena_frame
from com_protocol import build_claw_deliver, build_setspeed, build_turn, send_command
from id_color import color_blobs, robot_pose_approx
from map_utils import clamp_map_point, point_distance, robot_center_point
from robot_sync import (
    goto_map_point_with_pose,
    normalize_turn_angle,
    sync_robot_pose_from_camera,
)
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    DELIVERY_CENTER_TO_MARKER_DISTANCE,
    DELIVERY_GOAL_A_MARKER_FALLBACK,
    DELIVERY_GOAL_B_MARKER_FALLBACK,
    DELIVERY_GOAL_PREFERENCE,
    DELIVERY_HEADING_TOLERANCE,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    PICKUP_SETTLE_SECONDS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
)


def capture_delivery_scene_frame(camera):
    """Read one camera frame and detect robot pose plus both goal markers."""
    raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    color_matrix = create_matrix_from_frame(warped_frame)
    vision_scene = detect_vision_from_raw_frame(raw_frame)

    robot_pose = None
    if vision_scene is not None:
        robot_pose = vision_scene.robot_pose()

    if robot_pose is None and ALLOW_COLOR_DETECTION_FALLBACK:
        color_robot_pose = robot_pose_approx(color_matrix)

        if vision_scene is not None:
            robot_pose = vision_scene.robot_pose(fallback=color_robot_pose)

        if robot_pose is None:
            robot_pose = color_robot_pose

    goals = choose_delivery_goal_markers(color_matrix, vision_scene)

    return {
        "color_matrix": color_matrix,
        "vision_scene": vision_scene,
        "robot_pose": robot_pose,
        "goals": goals,
    }


def capture_delivery_scene(camera, retry_frames=ROBOT_POSE_RETRY_FRAMES):
    attempts = max(1, int(retry_frames))
    last_scene = None

    for attempt in range(1, attempts + 1):
        scene = capture_delivery_scene_frame(camera)
        last_scene = scene

        if scene is None:
            if attempt < attempts:
                print(
                    "Delivery camera: frame read failed; waiting for next frame ({}/{})".format(
                        attempt,
                        attempts,
                    )
                )
                time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)
            continue

        if scene["robot_pose"] is not None:
            if attempt > 1:
                print("Delivery camera: robot pose recovered on frame {}".format(attempt))
            return scene

        if attempt < attempts:
            print(
                "Delivery camera: robot pose missing; waiting for next frame ({}/{})".format(
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("Delivery camera: robot pose still missing after {} frames".format(attempts))
    return last_scene


def delivery_goal_marker_blob_is_valid(blob):
    return (
        6 <= blob["area"] <= 90
        and 5 <= blob["height"] <= 35
        and 2 <= blob["width"] <= 6
    )


def delivery_blob_center(blob):
    return int(round(blob["center_row"])), int(round(blob["center_col"]))


def choose_delivery_goal_markers(color_matrix, vision_scene=None):
    """Pick the most plausible pair of delivery markers."""
    if vision_scene is not None:
        vision_goals = vision_scene.goal_markers()

        if vision_goals is not None:
            goal_a, goal_b = vision_goals
            print(
                "Delivery marker detection: using vision Goal_A={}, Goal_B={}".format(
                    goal_a,
                    goal_b,
                )
            )
            return goal_a, goal_b

    if not ALLOW_COLOR_DETECTION_FALLBACK:
        print(
            "Delivery marker detection: vision did not find both goals and color fallback is disabled"
        )
        return None

    row_count = len(color_matrix)
    col_count = len(color_matrix[0])

    left_candidates = [
        blob for blob in color_blobs(color_matrix, "C", min_area=3, max_area=120)
        if blob["center_col"] < col_count * 0.45
        and delivery_goal_marker_blob_is_valid(blob)
    ]

    if left_candidates:
        left_blob = max(left_candidates, key=lambda blob: blob["area"])
        goal_b = delivery_blob_center(left_blob)
        left_row = left_blob["center_row"]
    else:
        goal_b = DELIVERY_GOAL_B_MARKER_FALLBACK
        left_row = float(goal_b[0])
        print("Delivery marker detection: using fallback Goal_B={}".format(goal_b))

    right_candidates = []

    for color in ("PK", "P"):
        for blob in color_blobs(color_matrix, color, min_area=3, max_area=120):
            if blob["center_col"] <= col_count * 0.55:
                continue
            if not delivery_goal_marker_blob_is_valid(blob):
                continue
            if abs(blob["center_row"] - left_row) > row_count * 0.25:
                continue

            right_candidates.append((color, blob))

    if not right_candidates:
        goal_a = (goal_b[0], DELIVERY_GOAL_A_MARKER_FALLBACK[1])
        print("Delivery marker detection: using fallback Goal_A={}".format(goal_a))
        return goal_a, goal_b

    right_color, right_blob = min(
        right_candidates,
        key=lambda item: (
            abs(item[1]["center_row"] - left_row),
            -item[1]["center_col"],
            0 if item[0] == "PK" else 1,
        ),
    )
    goal_a = delivery_blob_center(right_blob)

    print(
        "Delivery marker detection: Goal_A={} color={}, Goal_B={}".format(
            goal_a,
            right_color,
            goal_b,
        )
    )

    return goal_a, goal_b


def delivery_goal_heading(goal_name):
    if goal_name == "A":
        return 0.0
    if goal_name == "B":
        return 180.0
    raise ValueError("goal_name must be A or B")


def delivery_center_target(goal_marker, goal_name, center_to_marker_distance=DELIVERY_CENTER_TO_MARKER_DISTANCE):
    """Return the robot-center point that places the claw/ball at the goal marker."""
    marker_row, marker_col = goal_marker
    heading = delivery_goal_heading(goal_name)
    heading_rad = math.radians(heading)

    target_col = float(marker_col) - float(center_to_marker_distance) * math.cos(heading_rad)
    target_row = float(marker_row) - float(center_to_marker_distance) * math.sin(heading_rad)

    return clamp_map_point((target_row, target_col), margin=5)


def choose_delivery_goal(robot_pose, goal_a, goal_b, preference=DELIVERY_GOAL_PREFERENCE):
    """Pick a goal and return (name, marker)."""
    if preference == "A":
        return "A", goal_a
    if preference == "B":
        return "B", goal_b

    if robot_pose is None:
        return "A", goal_a

    robot_center = robot_center_point(robot_pose)
    target_a = delivery_center_target(goal_a, "A")
    target_b = delivery_center_target(goal_b, "B")

    if point_distance(robot_center, target_a) <= point_distance(robot_center, target_b):
        return "A", goal_a

    return "B", goal_b


def turn_delivery_to_heading(sock, camera, target_heading, tolerance_degrees=DELIVERY_HEADING_TOLERANCE):
    """Turn for delivery and verify from the camera before pushing the ball."""
    for attempt in range(1, 3):
        synced_pose = sync_robot_pose_from_camera(sock, camera)

        if synced_pose is None:
            return False

        _x, _y, current_heading = synced_pose
        turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))

        print(
            "Delivery heading attempt {}: current={:.1f}, target={:.1f}, turn={:.1f}".format(
                attempt,
                current_heading,
                target_heading,
                turn_angle,
            )
        )

        if abs(turn_angle) <= float(tolerance_degrees):
            return True

        if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
            return False

        time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)

    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return False

    _x, _y, current_heading = synced_pose
    final_error = normalize_turn_angle(float(target_heading) - float(current_heading))
    print("Delivery heading final error={:.1f}".format(final_error))

    return abs(final_error) <= float(tolerance_degrees)


def deliver_held_ball_to_goal(sock, camera):
    """Line up with one goal marker and run the EV3 delivery motion."""
    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery: could not read camera frame")
        return False

    robot_pose = scene["robot_pose"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery: could not detect robot pose")
        return False

    if goals is None:
        print("Delivery: could not detect both goal markers")
        return False

    goal_a, goal_b = goals
    goal_name, goal_marker = choose_delivery_goal(robot_pose, goal_a, goal_b)
    goal_heading = delivery_goal_heading(goal_name)

    delivery_waypoint = delivery_center_target(
        goal_marker,
        goal_name,
        DELIVERY_CENTER_TO_MARKER_DISTANCE,
    )

    print(
        "Delivery: chosen Goal_{} marker={}, waypoint={}, heading={:.1f}".format(
            goal_name,
            goal_marker,
            delivery_waypoint,
            goal_heading,
        )
    )

    if not goto_map_point_with_pose(sock, camera, robot_pose, delivery_waypoint, label="Delivery waypoint"):
        return False

    print("Delivery: aligning to goal heading {:.1f}".format(goal_heading))
    if not turn_delivery_to_heading(sock, camera, goal_heading):
        return False

    print("Delivery: stopping before deliver motion")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Delivery: running CLAW_DELIVER")
    return send_command(sock, build_claw_deliver())
