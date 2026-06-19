import time

from camera import (
    create_matrix_from_frame,
    detect_vision_from_warped_frame,
    read_arena_frame,
)
from id_color import ball_pos_approx_shape, grapler_pos_approx, robot_pose_approx
from path_obstacles import create_empty_path_matrix
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    ROBOT_POSE_RETRY_DELAY_SECONDS,
    ROBOT_POSE_RETRY_FRAMES,
)
from vision_debug_capture import save_missing_detection_frame


def capture_vision_scene_frame(
    camera,
    label,
    require_claw=True,
    require_robot_pose=True,
):
    """Capture one prepared arena frame with its color matrix and model scene."""
    _raw_frame, warped_frame = read_arena_frame(camera)

    if warped_frame is None:
        return None

    vision_scene = detect_vision_from_warped_frame(warped_frame)
    color_matrix = None

    if ALLOW_COLOR_DETECTION_FALLBACK:
        color_matrix = create_matrix_from_frame(warped_frame)

    save_missing_detection_frame(
        warped_frame,
        vision_scene,
        label,
        require_claw=require_claw,
        require_robot_pose=require_robot_pose,
    )

    return {
        "color_matrix": color_matrix,
        "path_matrix": create_empty_path_matrix(),
        "vision_scene": vision_scene,
        "warped_frame": warped_frame,
    }


def robot_pose_from_sources(color_matrix, vision_scene):
    """Use model pose first, then the configured color fallback if needed."""
    robot_pose = None

    if vision_scene is not None:
        robot_pose = vision_scene.robot_pose()

    if robot_pose is None and ALLOW_COLOR_DETECTION_FALLBACK and color_matrix is not None:
        color_robot_pose = robot_pose_approx(color_matrix)

        if vision_scene is not None:
            robot_pose = vision_scene.robot_pose(fallback=color_robot_pose)

        if robot_pose is None:
            robot_pose = color_robot_pose

    return robot_pose


def robot_body_visible(vision_scene):
    return vision_scene is not None and vision_scene.best("robot") is not None


def grappler_point_from_sources(color_matrix, vision_scene):
    """Use model claw/grappler point first, then color fallback if enabled."""
    grappler_point = None

    if vision_scene is not None:
        grappler_point = vision_scene.grappler_point()

    if grappler_point is None and ALLOW_COLOR_DETECTION_FALLBACK and color_matrix is not None:
        grappler_point = grapler_pos_approx(color_matrix, "G")

    return grappler_point


def ball_points_from_sources(color_matrix, vision_scene, ball_color):
    """Return detected ball centers for one project ball color."""
    balls = []

    if vision_scene is not None:
        balls = vision_scene.ball_points(ball_color)

    if not balls and ALLOW_COLOR_DETECTION_FALLBACK and color_matrix is not None:
        balls = ball_pos_approx_shape(color_matrix, ball_color)

    return balls


def capture_scene_with_robot_pose_retry(
    capture_frame,
    label,
    retry_frames=ROBOT_POSE_RETRY_FRAMES,
):
    """Retry a scene capture until robot_pose is present or attempts run out."""
    attempts = max(1, int(retry_frames))
    last_scene = None

    for attempt in range(1, attempts + 1):
        scene = capture_frame()
        last_scene = scene

        if scene is None:
            if attempt < attempts:
                print(
                    "{} camera: frame read failed; waiting for next frame ({}/{})".format(
                        label,
                        attempt,
                        attempts,
                    )
                )
                time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)
            continue

        if scene["robot_pose"] is not None:
            if attempt > 1:
                print("{} camera: robot pose recovered on frame {}".format(label, attempt))
            return scene

        if attempt < attempts:
            print(
                "{} camera: robot pose missing; waiting for next frame ({}/{})".format(
                    label,
                    attempt,
                    attempts,
                )
            )
            time.sleep(ROBOT_POSE_RETRY_DELAY_SECONDS)

    print("{} camera: robot pose still missing after {} frames".format(label, attempts))
    return last_scene
