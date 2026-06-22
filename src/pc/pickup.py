import math
import time

from collection_algorithm import A_star
from com_protocol import (
    build_claw_close,
    build_claw_open,
    build_goto,
    build_setspeed,
    send_command,
)
from map_utils import (
    clamp_map_point,
    heading_from_map_points,
    map_point_is_valid,
    path_is_valid,
    point_distance,
    robot_center_point,
    simplify_path_for_robot,
    truncate_path_before_target,
)
from path_obstacles import (
    clear_path_endpoint,
    clear_path_endpoint_preserving_obstacles,
    choose_safe_path_lookahead,
    clone_path_matrix,
    create_empty_path_matrix,
    mark_red_cross_obstacles,
    point_in_obstacle_regions,
    red_cross_obstacle_regions,
    segment_intersects_regions,
)
from robot_sync import (
    back_off_from_red_cross,
    goto_then_sync,
    goto_map_point_with_pose_pre_turn,
    map_xy_to_ev3_xy,
    normalize_turn_angle,
    reverse_for_missing_grappler,
    sync_robot_from_camera,
    sync_robot_pose_value,
    turn_robot_to_heading,
)
from scene_analysis import (
    ball_points_from_sources,
    capture_scene_with_robot_pose_retry,
    capture_vision_scene_frame,
    grappler_point_from_sources,
    robot_body_visible,
    robot_pose_from_sources,
)
from settings import (
    GRAPPLER_FORWARD_OFFSET_FALLBACK,
    GRAPPLER_LATERAL_OFFSET_FALLBACK,
    MAP_HEIGHT,
    MAP_WIDTH,
    PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
    PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE,
    PICKUP_CENTER_TO_BALL_MARGIN,
    PICKUP_FINAL_HEADING_CLOSE_TOLERANCE,
    PICKUP_FINAL_NUDGE_MARGIN,
    PICKUP_FINAL_NUDGE_MAX_DISTANCE,
    PICKUP_FINAL_SCOOP_DISTANCE,
    PICKUP_GRAPPLER_CLOSE_DISTANCE,
    PICKUP_OFFCENTER_DISTANCE_SCALE,
    PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT,
    PICKUP_PREAPPROACH_DISTANCE,
    PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    PICKUP_RED_CROSS_LOOKAHEAD_DISTANCE,
    PICKUP_SERVO_FAR_FORWARD_STEP,
    PICKUP_SERVO_MAX_FORWARD_STEP,
    PICKUP_SERVO_MAX_ITERATIONS,
    PICKUP_SERVO_MID_FORWARD_STEP,
    PICKUP_SERVO_MIN_FORWARD_STEP,
    PICKUP_SERVO_NEAR_FORWARD_STEP,
    PICKUP_SETTLE_SECONDS,
    PICKUP_STOP_DISTANCE,
    PICKUP_TARGET_MATCH_MAX_DISTANCE,
    PICKUP_WAYPOINT_STEP_SIZE,
    RED_CROSS_BACKOFF_MAX_ATTEMPTS,
    RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    ROBOT_POSE_RETRY_FRAMES,
    SYNC_DELAY_SECONDS,
    USE_COARSE_PICKUP_PREAPPROACH,
)
from vision_detection import set_vision_path_overlay


def planning_matrix_from_scene(scene):
    if scene is None:
        return create_empty_path_matrix()

    path_matrix = scene.get("path_matrix")

    if path_matrix is not None:
        return path_matrix

    color_matrix = scene.get("color_matrix")

    if color_matrix is not None:
        return color_matrix

    return create_empty_path_matrix()


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


def build_center_pickup_plan(
    color_matrix,
    grappler_to_ball_path,
    current_robot_pose,
    current_grappler_point,
):
    """
    Build a coarse pre-approach plan for the robot center.

    The final claw/ball geometry can change after the robot turns, so this only
    moves to a safe point near the ball. A camera-servo loop handles final pickup.
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


def follow_path_with_camera_sync(sock, camera, robot_path, step_size=10):
    if not path_is_valid(robot_path):
        print("Invalid path:", robot_path)
        return False

    if not sync_robot_from_camera(sock, camera):
        return False

    waypoints = simplify_path_for_robot(robot_path, min_spacing=step_size)

    if len(waypoints) > 1:
        waypoints = waypoints[1:]

    for row, col in waypoints:
        if not goto_then_sync(sock, camera, row, col):
            return False

    return True


def path_point_at_distance(robot_path, target_distance):
    if not path_is_valid(robot_path):
        return None

    if len(robot_path) == 1:
        return robot_path[0]

    target_distance = max(0.0, float(target_distance))
    travelled = 0.0

    for index in range(1, len(robot_path)):
        previous_point = robot_path[index - 1]
        current_point = robot_path[index]
        segment_length = point_distance(previous_point, current_point)

        if travelled + segment_length >= target_distance:
            if segment_length <= 0.001:
                return current_point

            ratio = (target_distance - travelled) / segment_length
            row = previous_point[0] + (current_point[0] - previous_point[0]) * ratio
            col = previous_point[1] + (current_point[1] - previous_point[1]) * ratio
            return int(round(row)), int(round(col))

        travelled += segment_length

    return robot_path[-1]


def red_cross_routed_pickup_step(scene, start_point, target_point, distance_map_units):
    if scene is None:
        return None

    base_matrix = planning_matrix_from_scene(scene)
    vision_scene = scene.get("vision_scene")
    regions = red_cross_obstacle_regions(
        base_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not segment_intersects_regions(start_point, target_point, regions):
        return None

    path_matrix = clone_path_matrix(base_matrix)
    mark_red_cross_obstacles(
        path_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )
    clear_path_endpoint(path_matrix, start_point, radius=10, value=".")
    clear_path_endpoint_preserving_obstacles(
        path_matrix,
        path_matrix,
        target_point,
        radius=PICKUP_BALL_ENDPOINT_CLEAR_RADIUS,
        value=".",
        blocked_regions=regions,
    )
    robot_path = A_star(path_matrix, start_point, target_point)

    if not path_is_valid(robot_path):
        print(
            "Pickup servo: red cross blocks direct step, but A* could not find a safe route: {}".format(
                robot_path,
            )
        )
        return False

    lookahead_distance = max(
        float(distance_map_units),
        float(PICKUP_RED_CROSS_LOOKAHEAD_DISTANCE),
    )
    step_target = choose_safe_path_lookahead(
        robot_path,
        start_point,
        regions,
        min_distance=max(
            float(distance_map_units),
            float(RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS),
        ),
        max_distance=lookahead_distance,
        acceptance_radius=RED_CROSS_WAYPOINT_ACCEPTANCE_RADIUS,
    )

    if step_target is None:
        step_target = path_point_at_distance(robot_path, distance_map_units)

        if (
            step_target is None
            or point_in_obstacle_regions(step_target, regions)
            or segment_intersects_regions(start_point, step_target, regions)
        ):
            print("Pickup servo: red cross route had no safe lookahead waypoint")
            return False

    set_vision_path_overlay(
        [
            {
                "points": robot_path,
                "label": "Pickup red-cross route",
                "color": (255, 0, 255),
            },
            {
                "points": [start_point, step_target],
                "label": "Pickup lookahead",
                "color": (0, 255, 255),
            },
        ],
        label="Pickup route",
    )

    print(
        "Pickup servo: red cross blocks direct step; routing lookahead via {} "
        "(lookahead {:.1f})".format(
            step_target,
            lookahead_distance,
        )
    )
    return step_target


def choose_closest_ball_to_grappler(balls, grappler_point):
    if not balls or grappler_point is None:
        return None

    return min(balls, key=lambda ball: point_distance(ball, grappler_point))


def choose_pickup_ball(balls, grappler_point, target_point=None):
    if not balls:
        return None, "none"

    if target_point is not None:
        matched_ball = min(balls, key=lambda ball: point_distance(ball, target_point))
        match_distance = point_distance(matched_ball, target_point)
        max_match_distance = max(0.0, float(PICKUP_TARGET_MATCH_MAX_DISTANCE))

        if match_distance <= max_match_distance:
            print(
                "Pickup camera: following selected target {}; matched visible ball {} "
                "(target error {:.1f})".format(
                    target_point,
                    matched_ball,
                    match_distance,
                )
            )
            return matched_ball, "target"

        print(
            "Pickup camera: selected target {} is not visible within {:.1f} map units; "
            "closest visible ball {} is {:.1f} away, so not switching targets".format(
                target_point,
                max_match_distance,
                matched_ball,
                match_distance,
            )
        )
        return None, "target_missing"

    return choose_closest_ball_to_grappler(balls, grappler_point), "closest"


def filter_balls_for_red_cross_clearance(color_matrix, vision_scene, balls):
    regions = red_cross_obstacle_regions(
        color_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not regions:
        return balls, []

    safe_balls = []
    blocked_balls = []

    for ball in balls:
        if point_in_obstacle_regions(ball, regions):
            print(
                "Pickup camera: ignoring ball {} because it is too close to the red cross".format(
                    ball,
                )
            )
            blocked_balls.append(ball)
            continue

        safe_balls.append(ball)

    return safe_balls, blocked_balls


def pickup_offcenter_ratio(point):
    if point is None:
        return 0.0

    row, col = point
    center_row = float(MAP_HEIGHT - 1) / 2.0
    center_col = float(MAP_WIDTH - 1) / 2.0
    max_distance = math.hypot(center_row, center_col)

    if max_distance <= 0.0:
        return 0.0

    distance = math.hypot(float(row) - center_row, float(col) - center_col)
    return max(0.0, min(1.0, distance / max_distance))


def pickup_distance_scale(point):
    return 1.0 + float(PICKUP_OFFCENTER_DISTANCE_SCALE) * pickup_offcenter_ratio(point)


def scaled_pickup_distance(distance, point):
    scale = pickup_distance_scale(point)
    return float(distance) * scale, scale


def scaled_final_scoop_distance(point):
    scale = min(
        float(PICKUP_OFFCENTER_SCOOP_SCALE_LIMIT),
        pickup_distance_scale(point),
    )
    return float(PICKUP_FINAL_SCOOP_DISTANCE) * scale


def capture_pickup_scene_frame(camera, ball_color="W", target_point=None):
    """Read one camera frame and return detection results used for pickup."""
    scene = capture_vision_scene_frame(camera, "pickup")

    if scene is None:
        return None

    color_matrix = scene["color_matrix"]
    path_matrix = scene["path_matrix"]
    vision_scene = scene["vision_scene"]
    robot_pose = robot_pose_from_sources(color_matrix, vision_scene)
    grappler_point = grappler_point_from_sources(color_matrix, vision_scene)
    balls = ball_points_from_sources(color_matrix, vision_scene, ball_color)

    balls, blocked_balls = filter_balls_for_red_cross_clearance(
        path_matrix,
        vision_scene,
        balls,
    )
    ball_point, ball_selection = choose_pickup_ball(
        balls,
        grappler_point,
        target_point=target_point,
    )

    return {
        "color_matrix": color_matrix,
        "path_matrix": path_matrix,
        "vision_scene": vision_scene,
        "robot_pose": robot_pose,
        "grappler_point": grappler_point,
        "balls": balls,
        "blocked_balls": blocked_balls,
        "pickup_blocked_by_red_cross": bool(blocked_balls and not balls),
        "pickup_target_point": target_point,
        "pickup_target_missing": ball_selection == "target_missing",
        "ball_selection": ball_selection,
        "ball_point": ball_point,
    }


def capture_pickup_scene(
    camera,
    ball_color="W",
    retry_frames=ROBOT_POSE_RETRY_FRAMES,
    target_point=None,
):
    return capture_scene_with_robot_pose_retry(
        lambda: capture_pickup_scene_frame(
            camera,
            ball_color=ball_color,
            target_point=target_point,
        ),
        "Pickup",
        retry_frames=retry_frames,
    )


def pickup_servo_forward_step(center_to_ball):
    """Use fewer, larger moves far away and smaller moves near the ball."""
    if center_to_ball > 140.0:
        return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_FAR_FORWARD_STEP)

    if center_to_ball > 80.0:
        return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_MID_FORWARD_STEP)

    return min(PICKUP_SERVO_MAX_FORWARD_STEP, PICKUP_SERVO_NEAR_FORWARD_STEP)


def drive_toward_map_point(sock, camera, robot_pose, target_point, distance_map_units, scene=None):
    """
    Move the robot center a short distance toward target_point.

    This deliberately sends one GOTO per camera frame.
    """
    center_x, center_y, _heading = robot_pose
    center_point = (int(round(center_y)), int(round(center_x)))
    distance_to_target = point_distance(center_point, target_point)

    if distance_to_target <= 0.001:
        print("Pickup servo: center is already at target point")
        return True

    distance_map_units = float(distance_map_units)
    routed_step = red_cross_routed_pickup_step(
        scene,
        center_point,
        target_point,
        distance_map_units,
    )

    if routed_step is False:
        regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene") if scene is not None else None,
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )

        if back_off_from_red_cross(
            sock,
            robot_pose=robot_pose,
            regions=regions,
            label="Pickup servo route recovery",
        ):
            return True

        return False

    if routed_step is not None:
        target_row, target_col = routed_step
    else:
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

    if routed_step is not None:
        return goto_map_point_with_pose_pre_turn(
            sock,
            camera,
            robot_pose,
            (target_row, target_col),
            label="Pickup red-cross routed step",
        )

    print(
        "Pickup servo: center->ball distance={:.1f}; moving {:.1f} to map center=({}, {})".format(
            distance_to_target,
            distance_map_units,
            target_col,
            target_row,
        )
    )
    set_vision_path_overlay(
        [center_point, (target_row, target_col), target_point],
        label="Pickup servo step",
        color=(0, 255, 255),
    )

    if not sync_robot_pose_value(sock, robot_pose, label="Pickup servo pre-GOTO"):
        return False

    ev3_x, ev3_y = map_xy_to_ev3_xy(target_col, target_row)
    print("Pickup servo: GOTO ev3=({}, {})".format(ev3_x, ev3_y))

    if not send_command(sock, build_goto(ev3_x, ev3_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def final_scoop_is_safe_from_red_cross(scene, robot_pose, target_point):
    if scene is None:
        return True

    base_matrix = planning_matrix_from_scene(scene)
    vision_scene = scene.get("vision_scene")
    regions = red_cross_obstacle_regions(
        base_matrix,
        vision_scene,
        margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
    )

    if not regions:
        return True

    center_x, center_y, _heading = robot_pose
    center_point = (int(round(center_y)), int(round(center_x)))
    grappler_point = scene.get("grappler_point")
    ball_point = scene.get("ball_point")

    for label, point in (
        ("robot center", center_point),
        ("grappler", grappler_point),
        ("ball", ball_point),
    ):
        if point_in_obstacle_regions(point, regions):
            print(
                "Pickup final scoop: refusing to move because {} {} is inside the red-cross pickup clearance".format(
                    label,
                    point,
                )
            )
            return False

    if grappler_point is not None and ball_point is not None:
        if segment_intersects_regions(grappler_point, ball_point, regions):
            print(
                "Pickup final scoop: refusing to close because the grappler-to-ball line crosses the red-cross pickup clearance"
            )
            return False

    if segment_intersects_regions(center_point, target_point, regions):
        print(
            "Pickup final scoop: refusing forward move because it crosses the red-cross pickup clearance"
        )
        return False

    return True


def final_scoop_forward_before_close(
    sock,
    camera,
    robot_pose,
    distance_map_units=PICKUP_FINAL_SCOOP_DISTANCE,
    scene=None,
):
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
    target_point = (target_y, target_x)

    if not final_scoop_is_safe_from_red_cross(scene, robot_pose, target_point):
        regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene") if scene is not None else None,
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )
        back_off_from_red_cross(
            sock,
            robot_pose=robot_pose,
            regions=regions,
            label="Pickup final scoop recovery",
        )
        return False

    print(
        "Pickup final scoop: moving forward {:.1f} map units to map center=({}, {}) before closing".format(
            distance_map_units,
            target_x,
            target_y,
        )
    )
    set_vision_path_overlay(
        [robot_center_point(robot_pose), target_point],
        label="Pickup final scoop",
        color=(0, 255, 255),
    )

    if not sync_robot_pose_value(sock, robot_pose, label="Pickup final scoop pre-GOTO"):
        return False

    ev3_x, ev3_y = map_xy_to_ev3_xy(target_x, target_y)
    print("Pickup final scoop: GOTO ev3=({}, {})".format(ev3_x, ev3_y))

    if not send_command(sock, build_goto(ev3_x, ev3_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def servo_align_and_approach_ball(sock, camera, ball_color="W", target_ball_point=None):
    """Final camera-servo pickup controller."""
    last_scene = None
    red_cross_backoff_count = 0

    for iteration in range(1, PICKUP_SERVO_MAX_ITERATIONS + 1):
        scene = capture_pickup_scene(
            camera,
            ball_color=ball_color,
            target_point=target_ball_point,
        )
        last_scene = scene

        if scene is None:
            print("Pickup servo: could not read camera frame")
            return False

        robot_pose = scene["robot_pose"]
        grappler_point = scene["grappler_point"]
        ball_point = scene["ball_point"]

        if grappler_point is None:
            if robot_body_visible(scene["vision_scene"]):
                if reverse_for_missing_grappler(sock, label="Pickup servo"):
                    continue
                return False

            print("Pickup servo: missing grappler and robot body detection")
            return False

        if robot_pose is None:
            print("Pickup servo: missing robot detection")
            return False

        red_cross_regions = red_cross_obstacle_regions(
            planning_matrix_from_scene(scene),
            scene.get("vision_scene"),
            margin=PICKUP_RED_CROSS_CLEARANCE_MARGIN,
        )
        servo_center_point = (int(round(robot_pose[1])), int(round(robot_pose[0])))
        robot_inside_clearance = point_in_obstacle_regions(servo_center_point, red_cross_regions)
        grappler_inside_clearance = point_in_obstacle_regions(grappler_point, red_cross_regions)

        if robot_inside_clearance or grappler_inside_clearance:
            if red_cross_backoff_count >= RED_CROSS_BACKOFF_MAX_ATTEMPTS:
                print(
                    "Pickup servo: still inside the red-cross clearance after {} backoff(s); "
                    "continuing escape instead of stopping".format(red_cross_backoff_count)
                )

            blocked_part = "robot center" if robot_inside_clearance else "claw"
            print(
                "Pickup servo: {} is inside the red-cross clearance; backing away "
                "instead of stopping (backoff {}/{})".format(
                    blocked_part,
                    red_cross_backoff_count + 1,
                    RED_CROSS_BACKOFF_MAX_ATTEMPTS,
                )
            )

            if not back_off_from_red_cross(
                sock,
                robot_pose=robot_pose,
                regions=red_cross_regions,
                label="Pickup servo",
            ):
                return False

            red_cross_backoff_count += 1
            continue

        red_cross_backoff_count = 0

        if ball_point is None:
            if scene.get("pickup_target_missing"):
                print(
                    "Pickup servo: selected ball target is not visible; retrying instead "
                    "of switching to a different nearby ball"
                )
                return False

            if scene.get("pickup_blocked_by_red_cross"):
                print(
                    "Pickup servo: only visible ball is inside the red-cross pickup clearance; not closing claw"
                )
                return False

            print(
                "Pickup servo: ball is no longer visible; assuming it is at/inside the claw and closing"
            )
            return True

        center_x, center_y, current_heading = robot_pose
        center_point = (int(round(center_y)), int(round(center_x)))
        center_to_ball_raw = point_distance(center_point, ball_point)
        center_to_ball, perspective_scale = scaled_pickup_distance(center_to_ball_raw, ball_point)
        final_scoop_distance = scaled_final_scoop_distance(ball_point)
        target_heading = heading_from_map_points(center_point, ball_point)
        heading_error = normalize_turn_angle(target_heading - current_heading)

        if grappler_point is not None:
            grappler_to_ball_raw = point_distance(grappler_point, ball_point)
            grappler_to_ball = grappler_to_ball_raw * perspective_scale
            grappler_text = (
                ", grappler={}, grappler_distance={:.1f}, "
                "scaled_grappler_distance={:.1f}"
            ).format(
                grappler_point,
                grappler_to_ball_raw,
                grappler_to_ball,
            )
        else:
            grappler_text = ", grappler=None"

        print(
            "Pickup servo iteration {}: center={}, ball={}, center_distance={:.1f}, "
            "scaled_center_distance={:.1f}, perspective_scale={:.2f}, "
            "heading={:.1f}, target_heading={:.1f}, heading_error={:.1f}{}".format(
                iteration,
                center_point,
                ball_point,
                center_to_ball_raw,
                center_to_ball,
                perspective_scale,
                current_heading,
                target_heading,
                heading_error,
                grappler_text,
            )
        )

        if grappler_point is not None and grappler_to_ball <= PICKUP_GRAPPLER_CLOSE_DISTANCE:
            print(
                "Pickup servo: ball is inside grappler range; doing final scoop instead of turning"
            )
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

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

                continue

            print("Pickup servo: center is close enough; doing final scoop before claw close")
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

        max_step = pickup_servo_forward_step(center_to_ball)
        forward_distance = center_to_ball - PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE
        forward_distance = min(forward_distance, max_step)

        if forward_distance < PICKUP_SERVO_MIN_FORWARD_STEP:
            print("Pickup servo: remaining center move is tiny; doing final scoop before claw close")
            return final_scoop_forward_before_close(
                sock,
                camera,
                robot_pose,
                distance_map_units=final_scoop_distance,
                scene=scene,
            )

        print(
            "Pickup servo step choice: max_step={:.1f}, requested_forward={:.1f}".format(
                max_step,
                forward_distance,
            )
        )

        if not drive_toward_map_point(
            sock,
            camera,
            robot_pose,
            ball_point,
            forward_distance,
            scene=scene,
        ):
            return False

    final_scene = capture_pickup_scene(
        camera,
        ball_color=ball_color,
        target_point=target_ball_point,
    )
    if final_scene is not None:
        if final_scene.get("pickup_target_missing"):
            print(
                "Pickup servo: final frame lost the selected ball target; not closing claw on a different ball"
            )
            return False

        if final_scene.get("pickup_blocked_by_red_cross"):
            print(
                "Pickup servo: final frame only sees a ball inside the red-cross pickup clearance; not closing claw"
            )
            return False

        final_robot = final_scene["robot_pose"]
        final_grappler = final_scene["grappler_point"]
        final_ball = final_scene["ball_point"]

        if final_grappler is None and robot_body_visible(final_scene["vision_scene"]):
            reverse_for_missing_grappler(sock, label="Pickup final frame")
            return False

        if final_robot is not None and final_ball is not None:
            final_center = (int(round(final_robot[1])), int(round(final_robot[0])))
            final_distance_raw = point_distance(final_center, final_ball)
            final_distance, final_scale = scaled_pickup_distance(final_distance_raw, final_ball)
            final_scoop_distance = scaled_final_scoop_distance(final_ball)
            print(
                "Pickup servo: max iterations reached; final center distance={:.1f}, "
                "scaled_final_distance={:.1f}, perspective_scale={:.2f}".format(
                    final_distance_raw,
                    final_distance,
                    final_scale,
                )
            )
            if final_distance > PICKUP_CENTER_TO_BALL_CLOSE_DISTANCE + 8.0:
                print("Pickup servo: still too far from ball; not closing claw")
                return False

            if final_robot is not None:
                return final_scoop_forward_before_close(
                    sock,
                    camera,
                    final_robot,
                    distance_map_units=final_scoop_distance,
                    scene=final_scene,
                )

    print("Pickup servo: max iterations reached but final distance is acceptable; closing")
    return True


def final_pickup_camera_nudge(
    sock,
    camera,
    ball_color="W",
    allow_extra_turn=True,
    target_ball_point=None,
):
    """
    Make one small camera-based forward correction after final alignment.

    This is deliberately conservative.
    """
    scene = capture_pickup_scene(
        camera,
        ball_color=ball_color,
        target_point=target_ball_point,
    )

    if scene is None:
        return False

    robot_pose = scene["robot_pose"]
    grappler_point = scene["grappler_point"]
    ball_point = scene["ball_point"]

    if grappler_point is None and robot_body_visible(scene["vision_scene"]):
        reverse_for_missing_grappler(sock, label="Pickup final nudge")
        return False

    if robot_pose is None or grappler_point is None or ball_point is None:
        print("Pickup final nudge: missing robot, grappler, or ball detection")
        return True

    ball_heading = heading_from_map_points(grappler_point, ball_point)
    _center_x, _center_y, current_heading = robot_pose
    heading_error = normalize_turn_angle(ball_heading - current_heading)
    distance_to_ball_raw = point_distance(grappler_point, ball_point)
    distance_to_ball, perspective_scale = scaled_pickup_distance(distance_to_ball_raw, ball_point)

    print(
        "Pickup final nudge: grappler={}, ball={}, distance={:.1f}, "
        "scaled_distance={:.1f}, perspective_scale={:.2f}, "
        "ball_heading={:.1f}, robot_heading={:.1f}, heading_error={:.1f}".format(
            grappler_point,
            ball_point,
            distance_to_ball_raw,
            distance_to_ball,
            perspective_scale,
            ball_heading,
            current_heading,
            heading_error,
        )
    )

    if abs(heading_error) > 12.0:
        if not allow_extra_turn:
            print("Pickup final nudge: heading still off after extra turn; not nudging")
            return True

        if not turn_robot_to_heading(sock, camera, ball_heading, tolerance_degrees=4.0):
            return False

        return final_pickup_camera_nudge(
            sock,
            camera,
            ball_color,
            allow_extra_turn=False,
            target_ball_point=target_ball_point,
        )

    center_x, center_y, current_heading = robot_pose

    desired_distance = float(PICKUP_STOP_DISTANCE)
    forward_distance = distance_to_ball - desired_distance

    if forward_distance <= PICKUP_FINAL_NUDGE_MARGIN:
        print("Pickup final nudge: distance is already good")
        return True

    forward_distance = min(forward_distance, float(PICKUP_FINAL_NUDGE_MAX_DISTANCE))
    heading_rad = math.radians(current_heading)
    target_x = int(round(float(center_x) + forward_distance * math.cos(heading_rad)))
    target_y = int(round(float(center_y) + forward_distance * math.sin(heading_rad)))

    if not (0 <= target_x < MAP_WIDTH and 0 <= target_y < MAP_HEIGHT):
        print("Pickup final nudge target is out of bounds:", (target_x, target_y))
        return True

    print(
        "Pickup final nudge: driving forward {:.1f} map units to map center=({}, {})".format(
            forward_distance,
            target_x,
            target_y,
        )
    )
    set_vision_path_overlay(
        [robot_center_point(robot_pose), (target_y, target_x)],
        label="Pickup final nudge",
        color=(0, 255, 255),
    )

    ev3_x, ev3_y = map_xy_to_ev3_xy(target_x, target_y)
    print("Pickup final nudge: GOTO ev3=({}, {})".format(ev3_x, ev3_y))

    if not send_command(sock, build_goto(ev3_x, ev3_y)):
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
    ball_color="W",
    open_claw=True,
    target_ball_point=None,
):
    """
    Open the claw, drive to a stable pre-approach point, then use fresh camera
    feedback for the final alignment and forward pickup motion.
    """
    center_pickup_path = None

    if USE_COARSE_PICKUP_PREAPPROACH:
        pickup_plan = build_center_pickup_plan(
            color_matrix,
            grappler_to_ball_path,
            current_robot_pose,
            current_grappler_point,
        )

        if pickup_plan is not None and path_is_valid(pickup_plan["center_path"]):
            center_pickup_path = pickup_plan["center_path"]
        else:
            print(
                "Pickup pre-approach plan failed; opening claw and using "
                "camera-servo pickup from the current position"
            )
    else:
        print(
            "Skipping coarse pickup pre-approach; using camera-servo approach "
            "from the current position"
        )

    if open_claw:
        print("Opening claw before pickup approach")
        if not send_command(sock, build_claw_open()):
            return False
    else:
        print("Continuing pickup without reopening claw")

    if center_pickup_path is not None:
        if not follow_path_with_camera_sync(
            sock,
            camera,
            center_pickup_path,
            step_size=PICKUP_WAYPOINT_STEP_SIZE,
        ):
            return False

    if not servo_align_and_approach_ball(
        sock,
        camera,
        ball_color=ball_color,
        target_ball_point=target_ball_point,
    ):
        print("Pickup servo failed; not closing claw blindly")
        return False

    print("Stopping before closing claw")
    if not send_command(sock, build_setspeed(0, 0)):
        return False

    time.sleep(PICKUP_SETTLE_SECONDS)

    print("Closing claw at pickup point")
    return send_command(sock, build_claw_close())
