import math
import time

from camera import create_matrix_from_frame, detect_vision_from_warped_frame, read_arena_frame
from collection_algorithm import A_star
from com_protocol import (
    build_claw_close,
    build_claw_deliver,
    build_setspeed,
    build_turn,
    send_command,
)
from id_color import color_blobs, robot_pose_approx
from map_utils import (
    clamp_map_point,
    path_is_valid,
    point_distance,
    robot_center_point,
    simplify_path_for_robot,
)
from path_obstacles import (
    clear_path_endpoint,
    clone_path_matrix,
    mark_red_cross_obstacles,
    red_cross_obstacle_regions,
    segment_intersects_regions,
)
from robot_sync import (
    goto_then_sync_with_pre_turn,
    goto_map_point_with_pose,
    normalize_turn_angle,
    sync_robot_pose_from_camera,
    sync_robot_pose_value,
)
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    DELIVERY_CENTER_TO_MARKER_DISTANCE,
    DELIVERY_CENTER_TO_CLAW_DISTANCE,
    DELIVERY_CLAW_EDGE_MARGIN,
    DELIVERY_CLAW_POSITION_TOLERANCE,
    DELIVERY_CLAW_TO_MARKER_DISTANCE,
    DELIVERY_EDGE_ESCAPE_REVERSE_SPEED,
    DELIVERY_EDGE_ESCAPE_SECONDS,
    DELIVERY_FINAL_CORRECTION_ATTEMPTS,
    DELIVERY_GOAL_A_MARKER_FALLBACK,
    DELIVERY_GOAL_B_MARKER_FALLBACK,
    DELIVERY_GOAL_PREFERENCE,
    DELIVERY_HEADING_TOLERANCE,
    DELIVERY_PREFER_VISION_GOALS,
    DELIVERY_POSITION_TOLERANCE,
    DELIVERY_REQUIRE_CENTER_POSITION,
    DELIVERY_ROBOT_EDGE_MARGIN,
    DELIVERY_USE_FIXED_GOALS,
    DELIVERY_WAYPOINT_STEP_SIZE,
    HELD_CLAW_RECLOSE_DELAY_SECONDS,
    HELD_CLAW_RECLOSE_ENABLED,
    MAP_HEIGHT,
    MAP_WIDTH,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    PICKUP_SETTLE_SECONDS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
)
from vision_debug_capture import save_missing_detection_frame


def capture_delivery_scene_frame(camera):
    """Read one camera frame and detect robot pose plus delivery goal positions."""
    _raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    color_matrix = create_matrix_from_frame(warped_frame)
    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(warped_frame, vision_scene, "delivery")

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
    grappler_point = None

    if vision_scene is not None:
        grappler_point = vision_scene.grappler_point()

    return {
        "color_matrix": color_matrix,
        "vision_scene": vision_scene,
        "robot_pose": robot_pose,
        "grappler_point": grappler_point,
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


def open_claw_detection_in_scene(scene):
    if scene is None:
        return None

    vision_scene = scene.get("vision_scene")

    if vision_scene is None:
        return None

    return vision_scene.open_claw_detection()


def ensure_held_claw_closed(sock, camera, scene, label="Delivery"):
    if not HELD_CLAW_RECLOSE_ENABLED:
        return scene

    detection = open_claw_detection_in_scene(scene)

    if detection is None:
        return scene

    print(
        "{}: vision sees open_claw at {} ({:.2f}) while claw should be closed; sending CLAW_CLOSE".format(
            label,
            detection.point,
            detection.confidence,
        )
    )

    if not send_command(sock, build_claw_close()):
        return None

    time.sleep(HELD_CLAW_RECLOSE_DELAY_SECONDS)

    refreshed_scene = capture_delivery_scene(camera)

    if refreshed_scene is None:
        print("{}: could not refresh camera after re-closing claw; continuing with previous frame".format(label))
        return scene

    refreshed_detection = open_claw_detection_in_scene(refreshed_scene)

    if refreshed_detection is not None:
        print(
            "{}: open_claw is still visible after close command at {} ({:.2f})".format(
                label,
                refreshed_detection.point,
                refreshed_detection.confidence,
            )
        )

    return refreshed_scene


def delivery_goal_marker_blob_is_valid(blob):
    return (
        6 <= blob["area"] <= 90
        and 5 <= blob["height"] <= 35
        and 2 <= blob["width"] <= 6
    )


def delivery_blob_center(blob):
    return int(round(blob["center_row"])), int(round(blob["center_col"]))


def fixed_delivery_goal_markers():
    return DELIVERY_GOAL_A_MARKER_FALLBACK, DELIVERY_GOAL_B_MARKER_FALLBACK


def choose_delivery_goal_markers(color_matrix, vision_scene=None):
    """Pick the most plausible pair of delivery markers."""
    if DELIVERY_USE_FIXED_GOALS and not DELIVERY_PREFER_VISION_GOALS:
        goal_a, goal_b = fixed_delivery_goal_markers()
        print(
            "Delivery marker detection: using fixed map openings Goal_A={}, Goal_B={}".format(
                goal_a,
                goal_b,
            )
        )
        return goal_a, goal_b

    if vision_scene is not None:
        vision_goals = vision_scene.goal_markers()

        if vision_goals is not None:
            goal_a, goal_b = vision_goals
            print(
                "Delivery marker detection: using vision openings Goal_A={}, Goal_B={}".format(
                    goal_a,
                    goal_b,
                )
            )
            return goal_a, goal_b

    if DELIVERY_USE_FIXED_GOALS:
        goal_a, goal_b = fixed_delivery_goal_markers()
        print(
            "Delivery marker detection: vision did not find both goals; using fixed map openings Goal_A={}, Goal_B={}".format(
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


def delivery_center_is_safe(point, margin=DELIVERY_ROBOT_EDGE_MARGIN):
    row, col = point
    return (
        margin <= row <= MAP_HEIGHT - 1 - margin
        and margin <= col <= MAP_WIDTH - 1 - margin
    )


def delivery_claw_is_clear_of_edge(point, margin=DELIVERY_CLAW_EDGE_MARGIN):
    if point is None:
        return False

    row, col = point
    return (
        margin <= row <= MAP_HEIGHT - 1 - margin
        and margin <= col <= MAP_WIDTH - 1 - margin
    )


def delivery_goal_marker_is_approachable(goal_marker, margin=DELIVERY_ROBOT_EDGE_MARGIN):
    marker_row, _marker_col = goal_marker
    return margin <= marker_row <= MAP_HEIGHT - 1 - margin


def clamp_delivery_center(point):
    clamped_point = clamp_map_point(point, margin=DELIVERY_ROBOT_EDGE_MARGIN)

    if clamped_point != (int(round(point[0])), int(round(point[1]))):
        print(
            "Delivery safety: clamped center target from {} to {}".format(
                (int(round(point[0])), int(round(point[1]))),
                clamped_point,
            )
        )

    return clamped_point


def delivery_claw_target(goal_marker, goal_name, claw_to_marker_distance=DELIVERY_CLAW_TO_MARKER_DISTANCE):
    """Return the claw point that is safely inside the field from the goal marker."""
    marker_row, marker_col = goal_marker
    heading = delivery_goal_heading(goal_name)
    heading_rad = math.radians(heading)

    target_col = float(marker_col) - float(claw_to_marker_distance) * math.cos(heading_rad)
    target_row = float(marker_row) - float(claw_to_marker_distance) * math.sin(heading_rad)

    return clamp_map_point((target_row, target_col), margin=5)


def delivery_center_target(
    goal_marker,
    goal_name,
    center_to_marker_distance=DELIVERY_CENTER_TO_MARKER_DISTANCE,
):
    """Return the robot-center point that should put the claw in front of the goal."""
    marker_row, marker_col = goal_marker
    heading = delivery_goal_heading(goal_name)
    heading_rad = math.radians(heading)
    center_to_claw_distance = DELIVERY_CENTER_TO_CLAW_DISTANCE

    if center_to_marker_distance != DELIVERY_CENTER_TO_MARKER_DISTANCE:
        center_to_claw_distance = max(
            0.0,
            float(center_to_marker_distance) - float(DELIVERY_CLAW_TO_MARKER_DISTANCE),
        )

    claw_row, claw_col = delivery_claw_target(goal_marker, goal_name)
    target_col = float(claw_col) - float(center_to_claw_distance) * math.cos(heading_rad)
    target_row = float(claw_row) - float(center_to_claw_distance) * math.sin(heading_rad)

    return clamp_delivery_center((target_row, target_col))


def delivery_center_target_from_claw_error(center, claw_point, claw_target):
    center_row, center_col = center
    claw_row, claw_col = claw_point
    target_row, target_col = claw_target

    corrected_center = (
        float(center_row) + float(target_row - claw_row),
        float(center_col) + float(target_col - claw_col),
    )

    return clamp_delivery_center(corrected_center)


def delivery_claw_points_toward_edge(claw_point, heading, margin=DELIVERY_CLAW_EDGE_MARGIN):
    if claw_point is None:
        return False

    row, col = claw_point
    heading_rad = math.radians(float(heading))
    forward_col = math.cos(heading_rad)
    forward_row = math.sin(heading_rad)

    return (
        (forward_col < -0.5 and col <= margin)
        or (forward_col > 0.5 and col >= MAP_WIDTH - 1 - margin)
        or (forward_row < -0.5 and row <= margin)
        or (forward_row > 0.5 and row >= MAP_HEIGHT - 1 - margin)
    )


def delivery_claw_is_on_goal_side(goal_name, marker, claw_point):
    marker_row, marker_col = marker
    claw_row, claw_col = claw_point
    heading_rad = math.radians(delivery_goal_heading(goal_name))
    forward_row = math.sin(heading_rad)
    forward_col = math.cos(heading_rad)
    marker_to_claw_backward = (
        (float(marker_row) - float(claw_row)) * forward_row
        + (float(marker_col) - float(claw_col)) * forward_col
    )

    return marker_to_claw_backward >= -float(DELIVERY_CLAW_POSITION_TOLERANCE)


def delivery_goal_option(goal_name, goal_marker):
    if not delivery_goal_marker_is_approachable(goal_marker):
        print(
            "Delivery safety: Goal_{} marker {} is too close to top/bottom wall; skipping".format(
                goal_name,
                goal_marker,
            )
        )
        return None

    claw_target = delivery_claw_target(goal_marker, goal_name)
    waypoint = delivery_center_target(goal_marker, goal_name)

    if not delivery_center_is_safe(waypoint):
        print(
            "Delivery safety: Goal_{} waypoint {} is not safe; skipping".format(
                goal_name,
                waypoint,
            )
        )
        return None

    return {
        "name": goal_name,
        "marker": goal_marker,
        "claw_target": claw_target,
        "waypoint": waypoint,
    }


def choose_delivery_goal(robot_pose, goal_a, goal_b, preference=DELIVERY_GOAL_PREFERENCE):
    """Pick a goal and return (name, marker)."""
    options = [
        option for option in (
            delivery_goal_option("A", goal_a),
            delivery_goal_option("B", goal_b),
        )
        if option is not None
    ]

    if preference in ("A", "B"):
        for option in options:
            if option["name"] == preference:
                return option

        print("Delivery: preferred Goal_{} is not safely approachable".format(preference))
        return None

    if robot_pose is None:
        if not options:
            return None

        return options[0]

    robot_center = robot_center_point(robot_pose)

    if not options:
        return None

    return min(options, key=lambda option: point_distance(robot_center, option["waypoint"]))


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


def delivery_path_matrix(scene, start_point, target_point):
    color_matrix = scene.get("color_matrix") if scene is not None else None

    if color_matrix is None:
        return None, 0

    path_matrix = clone_path_matrix(color_matrix)
    red_cross_count = mark_red_cross_obstacles(
        path_matrix,
        scene.get("vision_scene"),
    )

    if red_cross_count:
        clear_path_endpoint(path_matrix, start_point, radius=10, value=".")
        clear_path_endpoint(path_matrix, target_point, radius=10, value=".")

    return path_matrix, red_cross_count


def delivery_direct_path_crosses_red_cross(scene, start_point, target_point):
    color_matrix = scene.get("color_matrix") if scene is not None else None
    vision_scene = scene.get("vision_scene") if scene is not None else None
    regions = red_cross_obstacle_regions(color_matrix, vision_scene)

    if not regions:
        return False

    crosses_red_cross = segment_intersects_regions(start_point, target_point, regions)

    if not crosses_red_cross:
        print(
            "Delivery waypoint: direct segment {} -> {} does not cross red cross; using direct GOTO".format(
                start_point,
                target_point,
            )
        )

    return crosses_red_cross


def follow_delivery_path(sock, camera, robot_pose, robot_path, label):
    if not path_is_valid(robot_path):
        print("{}: invalid path {}".format(label, robot_path))
        return False

    if not sync_robot_pose_value(sock, robot_pose, label="{} pre-path".format(label)):
        return False

    waypoints = simplify_path_for_robot(
        robot_path,
        min_spacing=DELIVERY_WAYPOINT_STEP_SIZE,
    )

    if len(waypoints) > 1:
        waypoints = waypoints[1:]

    print("{}: following {} waypoint(s) around red cross".format(label, len(waypoints)))

    for index, (row, col) in enumerate(waypoints, start=1):
        print(
            "{} waypoint {}/{}: map=({}, {})".format(
                label,
                index,
                len(waypoints),
                col,
                row,
            )
        )

        waypoint_label = "{} waypoint {}/{}".format(label, index, len(waypoints))

        if not goto_then_sync_with_pre_turn(sock, camera, row, col, label=waypoint_label):
            return False

    return True


def goto_delivery_target(sock, camera, scene, robot_pose, target_point, label="Delivery waypoint"):
    if robot_pose is None:
        return goto_map_point_with_pose(sock, camera, robot_pose, target_point, label=label)

    start_point = robot_center_point(robot_pose)

    if point_distance(start_point, target_point) <= 2.0:
        print("{}: already at target {}".format(label, target_point))
        return sync_robot_pose_value(sock, robot_pose, label="{} already-at-target".format(label))

    if not delivery_direct_path_crosses_red_cross(scene, start_point, target_point):
        return goto_map_point_with_pose(sock, camera, robot_pose, target_point, label=label)

    path_matrix, red_cross_count = delivery_path_matrix(scene, start_point, target_point)

    if red_cross_count <= 0:
        return goto_map_point_with_pose(sock, camera, robot_pose, target_point, label=label)

    robot_path = A_star(path_matrix, start_point, target_point)

    if not path_is_valid(robot_path):
        print(
            "{}: red cross was detected, but A* could not find a safe route: {}".format(
                label,
                robot_path,
            )
        )
        return False

    print(
        "{}: planned red-cross avoiding route from {} to {}".format(
            label,
            start_point,
            target_point,
        )
    )
    return follow_delivery_path(sock, camera, robot_pose, robot_path, label)


def reverse_delivery_away_from_edge(sock, camera):
    print(
        "Delivery safety: claw is pointed into an edge; reversing for {:.2f}s before turning".format(
            DELIVERY_EDGE_ESCAPE_SECONDS,
        )
    )

    if not sync_robot_pose_from_camera(sock, camera):
        return None

    if not send_command(
        sock,
        build_setspeed(DELIVERY_EDGE_ESCAPE_REVERSE_SPEED, DELIVERY_EDGE_ESCAPE_REVERSE_SPEED),
    ):
        return None

    time.sleep(DELIVERY_EDGE_ESCAPE_SECONDS)

    if not send_command(sock, build_setspeed(0, 0)):
        return None

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)
    return sync_robot_pose_from_camera(sock, camera)


def move_to_safe_delivery_staging(sock, camera, scene):
    robot_pose = scene["robot_pose"]
    grappler_point = scene["grappler_point"]
    current_center = robot_center_point(robot_pose)
    safe_center = clamp_delivery_center(current_center)
    needs_staging = not delivery_center_is_safe(current_center) or safe_center != current_center

    if grappler_point is not None:
        _center_x, _center_y, current_heading = robot_pose

        if delivery_claw_points_toward_edge(grappler_point, current_heading):
            robot_pose = reverse_delivery_away_from_edge(sock, camera)

            if robot_pose is None:
                return None

            refreshed_scene = capture_delivery_scene(camera)

            if refreshed_scene is None or refreshed_scene["robot_pose"] is None:
                return robot_pose

            refreshed_scene = ensure_held_claw_closed(
                sock,
                camera,
                refreshed_scene,
                label="Delivery safety",
            )

            if refreshed_scene is None or refreshed_scene["robot_pose"] is None:
                return robot_pose

            scene = refreshed_scene
            robot_pose = scene["robot_pose"]
            grappler_point = scene["grappler_point"]
            current_center = robot_center_point(robot_pose)
            safe_center = clamp_delivery_center(current_center)
            needs_staging = (
                not delivery_center_is_safe(current_center)
                or safe_center != current_center
            )

        if grappler_point is not None and not delivery_claw_is_clear_of_edge(grappler_point):
            safe_grappler = clamp_map_point(grappler_point, margin=DELIVERY_CLAW_EDGE_MARGIN)
            delta_row = safe_grappler[0] - grappler_point[0]
            delta_col = safe_grappler[1] - grappler_point[1]
            safe_center = clamp_delivery_center(
                (
                    float(current_center[0]) + float(delta_row),
                    float(current_center[1]) + float(delta_col),
                )
            )
            needs_staging = True
            print(
                "Delivery safety: claw {} is near an edge; staging center to {}".format(
                    grappler_point,
                    safe_center,
                )
            )

    if not needs_staging:
        return robot_pose

    print(
        "Delivery safety: robot is near an edge at {}; staging inward to {}".format(
            current_center,
            safe_center,
        )
    )

    if not goto_delivery_target(sock, camera, scene, robot_pose, safe_center, label="Delivery safety staging"):
        return None

    return sync_robot_pose_from_camera(sock, camera)


def choose_fresh_delivery_option(sock, camera, goal_name):
    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery verify: could not read camera frame")
        return None, None, None

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery verify")

    if scene is None:
        print("Delivery verify: could not close claw after open_claw detection")
        return None, None, None

    robot_pose = scene["robot_pose"]
    grappler_point = scene["grappler_point"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery verify: could not detect robot pose")
        return None, None, None

    if goals is None:
        print("Delivery verify: could not detect both goal markers")
        return robot_pose, grappler_point, None

    goal_a, goal_b = goals
    goal_marker = goal_a if goal_name == "A" else goal_b
    return robot_pose, grappler_point, delivery_goal_option(goal_name, goal_marker)


def verify_delivery_alignment(sock, camera, goal_name, initial_option):
    option = initial_option

    for attempt in range(1, DELIVERY_FINAL_CORRECTION_ATTEMPTS + 2):
        robot_pose, grappler_point, fresh_option = choose_fresh_delivery_option(sock, camera, goal_name)

        if robot_pose is None:
            return False

        if grappler_point is None:
            print("Delivery verify: could not detect claw; not pushing")
            return False

        if fresh_option is None:
            print("Delivery verify: selected goal is not visible or not safely approachable")
            return False

        option = fresh_option

        center = robot_center_point(robot_pose)
        waypoint = option["waypoint"]
        marker = option["marker"]
        claw_target = option["claw_target"]
        goal_heading = delivery_goal_heading(goal_name)
        _x, _y, current_heading = robot_pose
        position_error = point_distance(center, waypoint)
        claw_error = point_distance(grappler_point, claw_target)
        heading_error = normalize_turn_angle(goal_heading - current_heading)
        claw_on_goal_side = delivery_claw_is_on_goal_side(goal_name, marker, grappler_point)
        position_ok = position_error <= DELIVERY_POSITION_TOLERANCE
        claw_ok = claw_error <= DELIVERY_CLAW_POSITION_TOLERANCE
        heading_ok = abs(heading_error) <= DELIVERY_HEADING_TOLERANCE
        center_position_required = bool(DELIVERY_REQUIRE_CENTER_POSITION)

        print(
            "Delivery verify attempt {}: center={}, claw={}, marker={}, "
            "claw_target={}, waypoint={}, position_error={:.1f}, "
            "claw_error={:.1f}, heading_error={:.1f}, position_ok={}, "
            "claw_ok={}, heading_ok={}, center_required={}, claw_on_goal_side={}".format(
                attempt,
                center,
                grappler_point,
                marker,
                claw_target,
                waypoint,
                position_error,
                claw_error,
                heading_error,
                position_ok,
                claw_ok,
                heading_ok,
                center_position_required,
                claw_on_goal_side,
            )
        )

        if not delivery_center_is_safe(center):
            print("Delivery safety: center {} is too close to a wall; not pushing".format(center))
            return False

        if not delivery_claw_is_clear_of_edge(grappler_point, margin=5):
            print("Delivery safety: claw {} is touching map edge; not pushing".format(grappler_point))
            return False

        if claw_ok and heading_ok and claw_on_goal_side and (position_ok or not center_position_required):
            if not position_ok:
                print(
                    "Delivery verify: accepting because claw is aligned; "
                    "center error {:.1f} is only a soft check".format(position_error)
                )
            return True

        if attempt > DELIVERY_FINAL_CORRECTION_ATTEMPTS:
            print("Delivery verify: still not lined up; not pushing")
            return False

        if not claw_ok:
            corrected_waypoint = delivery_center_target_from_claw_error(
                center,
                grappler_point,
                claw_target,
            )
            print(
                "Delivery verify: correcting claw position before push via center {}".format(
                    corrected_waypoint,
                )
            )
            if not goto_map_point_with_pose(
                sock,
                camera,
                robot_pose,
                corrected_waypoint,
                label="Delivery claw correction",
            ):
                return False
        elif center_position_required and not position_ok:
            print("Delivery verify: correcting center position before push")
            if not goto_map_point_with_pose(
                sock,
                camera,
                robot_pose,
                waypoint,
                label="Delivery final correction",
            ):
                return False

        if not heading_ok:
            print("Delivery verify: correcting heading before push")
            if not turn_delivery_to_heading(sock, camera, goal_heading):
                return False

    return False


def deliver_held_ball_to_goal(sock, camera):
    """Line up with one goal marker and run the EV3 delivery motion."""
    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery: could not read camera frame")
        return False

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery start")

    if scene is None:
        print("Delivery: could not close claw after open_claw detection")
        return False

    robot_pose = scene["robot_pose"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery: could not detect robot pose")
        return False

    if goals is None:
        print("Delivery: could not detect both goal markers")
        return False

    robot_pose = move_to_safe_delivery_staging(sock, camera, scene)

    if robot_pose is None:
        print("Delivery: could not move safely away from edge before scoring")
        return False

    scene = capture_delivery_scene(camera)

    if scene is None:
        print("Delivery: could not refresh camera frame after safety staging")
        return False

    scene = ensure_held_claw_closed(sock, camera, scene, label="Delivery after safety staging")

    if scene is None:
        print("Delivery: could not close claw after safety staging")
        return False

    robot_pose = scene["robot_pose"]
    goals = scene["goals"]

    if robot_pose is None:
        print("Delivery: could not detect robot pose after safety staging")
        return False

    if goals is None:
        print("Delivery: could not detect both goal markers after safety staging")
        return False

    goal_a, goal_b = goals
    delivery_option = choose_delivery_goal(robot_pose, goal_a, goal_b)

    if delivery_option is None:
        print("Delivery: no safely approachable goal")
        return False

    goal_name = delivery_option["name"]
    goal_marker = delivery_option["marker"]
    claw_target = delivery_option["claw_target"]
    delivery_waypoint = delivery_option["waypoint"]
    goal_heading = delivery_goal_heading(goal_name)

    print(
        "Delivery: chosen Goal_{} marker={}, claw_target={}, waypoint={}, heading={:.1f}".format(
            goal_name,
            goal_marker,
            claw_target,
            delivery_waypoint,
            goal_heading,
        )
    )

    if not goto_delivery_target(sock, camera, scene, robot_pose, delivery_waypoint, label="Delivery waypoint"):
        return False

    print("Delivery: aligning to goal heading {:.1f}".format(goal_heading))
    if not turn_delivery_to_heading(sock, camera, goal_heading):
        return False

    if not verify_delivery_alignment(sock, camera, goal_name, delivery_option):
        return False

    print("Delivery: stopping before deliver motion")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Delivery: running CLAW_DELIVER")
    return send_command(sock, build_claw_deliver())
