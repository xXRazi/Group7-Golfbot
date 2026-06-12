import time

from camera import detect_vision_from_raw_frame, read_arena_frame, save_frame
from com_protocol import build_goto, build_possync, build_turn, send_command
from id_color import robot_pose_approx
from Imagesplitter import create_matrix
from map_utils import clamp_map_point
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    PICKUP_FINAL_HEADING_TOLERANCE,
    PICKUP_FINAL_SYNC_DELAY_SECONDS,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
    SYNC_DELAY_SECONDS,
    SYNC_IMAGE_PATH,
)


def count_color(matrix, color):
    count = 0

    for row in matrix:
        for value in row:
            if value == color:
                count += 1

    return count


def get_robot_pose_from_camera_frame(camera):
    raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    save_frame(warped_frame, SYNC_IMAGE_PATH)
    print("Saved sync image:", SYNC_IMAGE_PATH)

    color_robot_pose = None

    if ALLOW_COLOR_DETECTION_FALLBACK:
        color_matrix = create_matrix(SYNC_IMAGE_PATH)

        print(
            "Robot marker counts: Y={}, P={}, B={}".format(
                count_color(color_matrix, "Y"),
                count_color(color_matrix, "P"),
                count_color(color_matrix, "B"),
            )
        )

        color_robot_pose = robot_pose_approx(color_matrix)
    vision_scene = detect_vision_from_raw_frame(raw_frame)

    if vision_scene is None:
        if not ALLOW_COLOR_DETECTION_FALLBACK:
            print("Camera sync: vision unavailable and color fallback is disabled")
        return color_robot_pose

    robot_pose = vision_scene.robot_pose(fallback=color_robot_pose)

    if robot_pose is not None:
        print("Camera sync vision pose:", robot_pose)
        return robot_pose

    if not ALLOW_COLOR_DETECTION_FALLBACK:
        print("Camera sync: vision did not detect a full robot pose and color fallback is disabled")

    return color_robot_pose


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


def goto_map_point_with_pose(sock, camera, robot_pose, target_point, label="Delivery GOTO"):
    """Sync a known camera pose, GOTO a map point, then sync again."""
    target_row, target_col = clamp_map_point(target_point, margin=5)

    print("{}: target center=({}, {})".format(label, target_col, target_row))

    if robot_pose is not None:
        if not sync_robot_pose_value(sock, robot_pose, label="{} pre-GOTO".format(label)):
            return False
    else:
        if not sync_robot_from_camera(sock, camera):
            return False

    if not send_command(sock, build_goto(target_col, target_row)):
        return False

    time.sleep(SYNC_DELAY_SECONDS)
    return sync_robot_from_camera(sock, camera)
