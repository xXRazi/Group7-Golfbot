import time

from camera import detect_vision_from_warped_frame, read_arena_frame, save_frame
from com_protocol import build_goto, build_possync, build_setspeed, build_turn, send_command
from Imagesplitter import create_matrix
from map_utils import clamp_map_point, heading_from_map_points, point_distance, robot_center_point
from scene_analysis import robot_pose_from_sources
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    EV3_MAP_HEIGHT,
    EV3_MAP_WIDTH,
    MAP_HEIGHT,
    MAP_WIDTH,
    MISSING_GRAPPLER_REVERSE_ENABLED,
    MISSING_GRAPPLER_REVERSE_SECONDS,
    MISSING_GRAPPLER_REVERSE_SETTLE_SECONDS,
    MISSING_GRAPPLER_REVERSE_SPEED,
    PATH_PRETURN_HEADING_TOLERANCE,
    PICKUP_FINAL_HEADING_TOLERANCE,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
    SYNC_DELAY_SECONDS,
    SYNC_IMAGE_PATH,
)
from vision_debug_capture import save_missing_detection_frame


_EV3_COORDINATE_WARNING_PRINTED = False


def _scale_axis(value, source_size, target_size):
    value = float(value)
    source_size = int(source_size)
    target_size = int(target_size)

    if source_size == target_size or source_size <= 1 or target_size <= 1:
        return int(round(value))

    return int(round(value * float(target_size - 1) / float(source_size - 1)))


def _print_ev3_coordinate_warning_once():
    global _EV3_COORDINATE_WARNING_PRINTED

    if _EV3_COORDINATE_WARNING_PRINTED:
        return

    if EV3_MAP_WIDTH != MAP_WIDTH or EV3_MAP_HEIGHT != MAP_HEIGHT:
        print(
            "EV3 coordinate scaling enabled: PC map={}x{}, EV3 map={}x{}".format(
                MAP_WIDTH,
                MAP_HEIGHT,
                EV3_MAP_WIDTH,
                EV3_MAP_HEIGHT,
            )
        )

    _EV3_COORDINATE_WARNING_PRINTED = True


def map_xy_to_ev3_xy(x, y):
    _print_ev3_coordinate_warning_once()
    ev3_x = _scale_axis(x, MAP_WIDTH, EV3_MAP_WIDTH)
    ev3_y = _scale_axis(y, MAP_HEIGHT, EV3_MAP_HEIGHT)
    return ev3_x, ev3_y


def map_point_to_ev3_xy(point):
    row, col = point
    return map_xy_to_ev3_xy(col, row)


def map_pose_to_ev3_pose(robot_pose):
    x, y, heading = robot_pose
    ev3_x, ev3_y = map_xy_to_ev3_xy(x, y)
    return ev3_x, ev3_y, heading


def ev3_xy_is_valid(x, y):
    return 0 <= int(round(x)) < EV3_MAP_WIDTH and 0 <= int(round(y)) < EV3_MAP_HEIGHT


def count_color(matrix, color):
    count = 0

    for row in matrix:
        for value in row:
            if value == color:
                count += 1

    return count


def get_robot_pose_from_camera_frame(camera):
    _raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    save_frame(warped_frame, SYNC_IMAGE_PATH)
    print("Saved sync image:", SYNC_IMAGE_PATH)

    color_matrix = None

    if ALLOW_COLOR_DETECTION_FALLBACK:
        color_matrix = create_matrix(SYNC_IMAGE_PATH)

        print(
            "Robot marker counts: Y={}, P={}, B={}".format(
                count_color(color_matrix, "Y"),
                count_color(color_matrix, "P"),
                count_color(color_matrix, "B"),
            )
        )

    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(
        warped_frame,
        vision_scene,
        "sync",
        require_claw=False,
        require_robot_pose=True,
    )

    if vision_scene is None:
        if not ALLOW_COLOR_DETECTION_FALLBACK:
            print("Camera sync: vision unavailable and color fallback is disabled")
        return robot_pose_from_sources(color_matrix, vision_scene)

    robot_pose = robot_pose_from_sources(color_matrix, vision_scene)

    if robot_pose is not None:
        print("Camera sync vision pose:", robot_pose)
        return robot_pose

    if not ALLOW_COLOR_DETECTION_FALLBACK:
        print("Camera sync: vision did not detect a full robot pose and color fallback is disabled")

    return None


def get_robot_pose_from_camera(camera, retry_frames=ROBOT_POSE_RETRY_FRAMES):
    attempts = max(1, int(retry_frames))

    for attempt in range(1, attempts + 1):
        pose = get_robot_pose_from_camera_frame(camera)

        if pose is not None:
            if attempt > 1:
                print("Camera sync: robot pose recovered on frame {}".format(attempt))
            return pose

        if attempt < attempts:
            print(
                "Camera sync: robot pose missing; waiting for next frame ({}/{})".format(
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("Camera sync: robot pose still missing after {} frames".format(attempts))
    return None


def sync_robot_pose_from_camera(sock, camera):
    pose = get_robot_pose_from_camera(camera)

    if pose is None:
        print("Could not detect robot pose from camera")
        return None

    map_x, map_y, heading = pose
    x, y = map_xy_to_ev3_xy(map_x, map_y)
    heading_tenths = int(round(heading * 10))

    print(
        "Camera sync: map=({}, {}), ev3=({}, {}), heading={:.1f}".format(
            int(round(map_x)),
            int(round(map_y)),
            x,
            y,
            heading,
        )
    )

    if not send_command(sock, build_possync(x, y, heading_tenths)):
        return None

    return int(round(map_x)), int(round(map_y)), heading


def sync_robot_from_camera(sock, camera):
    return sync_robot_pose_from_camera(sock, camera) is not None


def sync_robot_pose_value(sock, robot_pose, label="Camera pose"):
    """Send an already-detected camera pose to the EV3 odometry state."""
    if robot_pose is None:
        return False

    map_x, map_y, heading = robot_pose
    x, y = map_xy_to_ev3_xy(map_x, map_y)
    heading_tenths = int(round(float(heading) * 10))

    print(
        "{} sync: map=({}, {}), ev3=({}, {}), heading={:.1f}".format(
            label,
            int(round(map_x)),
            int(round(map_y)),
            x,
            y,
            heading,
        )
    )
    return send_command(sock, build_possync(x, y, heading_tenths))


def normalize_turn_angle(angle):
    """Normalize a heading correction to the shortest signed turn."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def reverse_for_missing_grappler(sock, label="Missing grappler"):
    if not MISSING_GRAPPLER_REVERSE_ENABLED:
        print("{}: missing grappler reverse is disabled".format(label))
        return False

    speed = -abs(int(round(MISSING_GRAPPLER_REVERSE_SPEED)))
    reverse_seconds = max(0.0, float(MISSING_GRAPPLER_REVERSE_SECONDS))
    settle_seconds = max(0.0, float(MISSING_GRAPPLER_REVERSE_SETTLE_SECONDS))

    if reverse_seconds <= 0.0 or speed == 0:
        print("{}: missing grappler reverse has no movement configured".format(label))
        return False

    print(
        "{}: grappler/claw is not visible; reversing at speed {} for {:.2f}s".format(
            label,
            speed,
            reverse_seconds,
        )
    )

    if not send_command(sock, build_setspeed(speed, speed)):
        return False

    time.sleep(reverse_seconds)

    if not send_command(sock, build_setspeed(0, 0)):
        return False

    if settle_seconds > 0.0:
        time.sleep(settle_seconds)

    return True


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
    map_x = int(round(col))
    map_y = int(round(row))
    x, y = map_xy_to_ev3_xy(map_x, map_y)

    print("Sending GOTO map=({}, {}), ev3=({}, {})".format(map_x, map_y, x, y))

    if not send_command(sock, build_goto(x, y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)

    return sync_robot_from_camera(sock, camera)


def goto_map_point_with_pose(sock, camera, robot_pose, target_point, label="Delivery GOTO"):
    """Sync a known camera pose, GOTO a map point, then sync again."""
    target_row, target_col = clamp_map_point(target_point, margin=5)
    target_x, target_y = map_point_to_ev3_xy((target_row, target_col))

    print(
        "{}: target center map=({}, {}), ev3=({}, {})".format(
            label,
            target_col,
            target_row,
            target_x,
            target_y,
        )
    )

    if robot_pose is not None:
        if not sync_robot_pose_value(sock, robot_pose, label="{} pre-GOTO".format(label)):
            return False
    else:
        if not sync_robot_from_camera(sock, camera):
            return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def _pre_turn_for_goto(sock, camera, robot_pose, target_point, label, tolerance_degrees):
    start_point = robot_center_point(robot_pose)

    if point_distance(start_point, target_point) <= 2.0:
        print("{} pre-turn: already at target".format(label))
        return robot_pose

    target_heading = heading_from_map_points(start_point, target_point)
    _map_x, _map_y, current_heading = robot_pose
    turn_angle = normalize_turn_angle(float(target_heading) - float(current_heading))
    tolerance_degrees = float(tolerance_degrees)

    print(
        "{} pre-turn: center={}, target={}, current_heading={:.1f}, "
        "target_heading={:.1f}, turn={:.1f}, tolerance={:.1f}".format(
            label,
            start_point,
            target_point,
            current_heading,
            target_heading,
            turn_angle,
            tolerance_degrees,
        )
    )

    if abs(turn_angle) <= tolerance_degrees:
        return robot_pose

    if not send_command(sock, build_turn(int(round(turn_angle)), 0)):
        return None

    time.sleep(PICKUP_FINAL_SYNC_DELAY_SECONDS)
    synced_pose = sync_robot_pose_from_camera(sock, camera)

    if synced_pose is None:
        return None

    _synced_x, _synced_y, synced_heading = synced_pose
    final_error = normalize_turn_angle(float(target_heading) - float(synced_heading))

    print(
        "{} pre-turn: verified_heading={:.1f}, error={:.1f}".format(
            label,
            synced_heading,
            final_error,
        )
    )

    if abs(final_error) > tolerance_degrees:
        print(
            "{} pre-turn: heading is still outside tolerance; refusing forward GOTO".format(
                label,
            )
        )
        return None

    return synced_pose


def goto_map_point_with_pose_pre_turn(
    sock,
    camera,
    robot_pose,
    target_point,
    label="Path GOTO",
    tolerance_degrees=PATH_PRETURN_HEADING_TOLERANCE,
):
    """Sync pose, rotate to the segment heading, then send the forward GOTO."""
    target_row, target_col = clamp_map_point(target_point, margin=5)
    target_point = (target_row, target_col)
    target_x, target_y = map_point_to_ev3_xy(target_point)

    print(
        "{}: target center map=({}, {}), ev3=({}, {})".format(
            label,
            target_col,
            target_row,
            target_x,
            target_y,
        )
    )

    if robot_pose is not None:
        if not sync_robot_pose_value(sock, robot_pose, label="{} pre-GOTO".format(label)):
            return False
        pose = robot_pose
    else:
        pose = sync_robot_pose_from_camera(sock, camera)

        if pose is None:
            return False

    pose = _pre_turn_for_goto(
        sock,
        camera,
        pose,
        target_point,
        label,
        tolerance_degrees,
    )

    if pose is None:
        return False

    if not send_command(sock, build_goto(target_x, target_y)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)


def goto_then_sync_with_pre_turn(
    sock,
    camera,
    row,
    col,
    label="Path GOTO",
    tolerance_degrees=PATH_PRETURN_HEADING_TOLERANCE,
):
    return goto_map_point_with_pose_pre_turn(
        sock,
        camera,
        None,
        (row, col),
        label=label,
        tolerance_degrees=tolerance_degrees,
    )
