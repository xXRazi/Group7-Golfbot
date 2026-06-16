#!/usr/bin/env python3
import os
import socket
import time
from dataclasses import dataclass

import cv2 as cv

from camera import (
    close_camera,
    detect_vision_from_warped_frame,
    ensure_image_dir,
    open_camera,
    save_frame,
    warp_frame,
)
from collection_algorithm import A_star, get_h_list
from com_protocol import HOST, PORT, build_handshake, build_mapsize, send_command
from delivery import deliver_held_ball_to_goal
from id_color import ball_pos_approx_shape, goals_pos_approx, grapler_pos_approx, robot_pose_approx
from Imagesplitter import create_matrix
from path_obstacles import clear_path_endpoint, mark_red_cross_obstacles
from pickup import approach_ball_and_close_claw
from settings import (
    ALLOW_COLOR_DETECTION_FALLBACK,
    CAMERA_INDEX,
    EV3_MAP_HEIGHT,
    EV3_MAP_WIDTH,
    FRAME_CAPTURE_INTERVAL_SECONDS,
    IMAGE_DIR,
    PICKUP_BALL_COLORS,
    STARTUP_DELAY_SECONDS,
    STOP_AFTER_SUCCESSFUL_DELIVERY,
)
from vision_debug_capture import save_missing_detection_frame
from vision_detection import vision_live_view_quit_requested


@dataclass
class AutonomousState:
    image_count: int = 0
    path_executed: bool = False
    pickup_started: bool = False
    begin_time: float = 0.0
    last_capture_time: float = 0.0


def startup_delay_has_elapsed(state, now):
    return now - state.begin_time >= STARTUP_DELAY_SECONDS


def capture_interval_has_elapsed(state, now):
    if now - state.last_capture_time < FRAME_CAPTURE_INTERVAL_SECONDS:
        return False

    state.last_capture_time = now
    return True


def capture_color_matrix(state, warped_frame):
    image_name = f"{state.image_count}.png"
    full_path = os.path.join(IMAGE_DIR, image_name)
    save_frame(warped_frame, full_path)
    state.image_count += 1

    print("Vi tager et billede")
    return create_matrix(full_path)


def capture_detection_scene(state, warped_frame):
    color_matrix = capture_color_matrix(state, warped_frame)
    vision_scene = detect_vision_from_warped_frame(warped_frame)
    save_missing_detection_frame(warped_frame, vision_scene, "main")

    return {
        "color_matrix": color_matrix,
        "vision_scene": vision_scene,
    }


def prepare_pickup_path_matrix(color_matrix, grapler_point, ball_point, vision_scene=None):
    mark_red_cross_obstacles(color_matrix, vision_scene)
    clear_path_endpoint(color_matrix, grapler_point, radius=8, value=".")
    clear_path_endpoint(color_matrix, ball_point, radius=5, value="W")
    return color_matrix


def sort_balls_by_distance_to_grappler(ball_targets, grapler_point):
    paired_targets = []

    for target in ball_targets:
        point = target["point"]
        distance = get_h_list(grapler_point[0], grapler_point[1], point[0], point[1])[0]
        paired_targets.append((distance, target))

    paired_targets.sort(key=lambda item: item[0])
    sorted_targets = [target for _distance, target in paired_targets]

    print(
        "Pickup ball targets by distance:",
        [
            "{}@{}:{:.1f}".format(target["color"], target["point"], distance)
            for distance, target in paired_targets
        ],
    )
    return sorted_targets


def detect_ball_points_for_color(color_matrix, vision_scene, ball_color):
    ball_points = []

    if vision_scene is not None:
        ball_points = vision_scene.ball_points(ball_color)

        if ball_points:
            print("Vision {} balls: {}".format(ball_color, ball_points))

    if not ball_points and ALLOW_COLOR_DETECTION_FALLBACK:
        ball_points = ball_pos_approx_shape(color_matrix, ball_color)

        if ball_points:
            print("Color fallback {} balls: {}".format(ball_color, ball_points))

    return ball_points


def detect_pickup_ball_targets(color_matrix, vision_scene=None):
    ball_targets = []

    for ball_color in PICKUP_BALL_COLORS:
        for point in detect_ball_points_for_color(color_matrix, vision_scene, ball_color):
            ball_targets.append(
                {
                    "color": ball_color,
                    "point": point,
                }
            )

    return ball_targets


def detect_pickup_target(color_matrix, vision_scene=None):
    ball_targets = detect_pickup_ball_targets(color_matrix, vision_scene)

    grapler_point = None

    if vision_scene is not None:
        grapler_point = vision_scene.grappler_point()

        if grapler_point is not None:
            print("Vision grapler:", grapler_point)

    if grapler_point is None and ALLOW_COLOR_DETECTION_FALLBACK:
        grapler_point = grapler_pos_approx(color_matrix, "G")

    print(grapler_point)

    if grapler_point is None:
        print("No grapler detected; cannot collect ball")
        return None

    current_robot_pose = None

    if vision_scene is not None:
        current_robot_pose = vision_scene.robot_pose()

        if current_robot_pose is not None:
            print("Vision robot pose:", current_robot_pose)

    if current_robot_pose is None and ALLOW_COLOR_DETECTION_FALLBACK:
        color_robot_pose = robot_pose_approx(color_matrix)

        if vision_scene is not None:
            current_robot_pose = vision_scene.robot_pose(fallback=color_robot_pose)

            if current_robot_pose is not None and current_robot_pose != color_robot_pose:
                print("Vision robot center with color heading:", current_robot_pose)

        if current_robot_pose is None:
            current_robot_pose = color_robot_pose

    if current_robot_pose is None:
        print("No robot pose detected; cannot collect ball")
        return None

    if not ball_targets:
        print("No pickup balls detected for colors {}".format(PICKUP_BALL_COLORS))
        return None

    ball_targets = sort_balls_by_distance_to_grappler(ball_targets, grapler_point)
    selected_ball = ball_targets[0]
    selected_ball_point = selected_ball["point"]
    selected_ball_color = selected_ball["color"]
    print(
        "Selected pickup target: color={}, point={}".format(
            selected_ball_color,
            selected_ball_point,
        )
    )
    pickup_matrix = prepare_pickup_path_matrix(
        color_matrix,
        grapler_point,
        selected_ball_point,
        vision_scene,
    )
    robot_path = A_star(pickup_matrix, grapler_point, selected_ball_point)

    return {
        "grapler_point": grapler_point,
        "robot_pose": current_robot_pose,
        "robot_path": robot_path,
        "ball_color": selected_ball_color,
        "ball_point": selected_ball_point,
    }


def handle_pickup_and_delivery(sock, camera, state, color_matrix, pickup_target):
    if state.path_executed:
        return False

    pickup_success = approach_ball_and_close_claw(
        sock,
        camera,
        color_matrix,
        pickup_target["robot_path"],
        current_grappler_point=pickup_target["grapler_point"],
        current_robot_pose=pickup_target["robot_pose"],
        ball_color=pickup_target["ball_color"],
        open_claw=not state.pickup_started,
    )

    state.pickup_started = True

    if not pickup_success:
        print(
            "Pickup attempt did not finish cleanly; will retry from the current position without reopening the claw"
        )
        return False

    print("Pickup succeeded; starting delivery")
    state.path_executed = True
    state.pickup_started = False

    return retry_delivery(sock, camera, state)


def retry_delivery(sock, camera, state):
    delivery_success = deliver_held_ball_to_goal(sock, camera)

    if delivery_success:
        print("Delivery complete")

        if STOP_AFTER_SUCCESSFUL_DELIVERY:
            return True

        state.path_executed = False
        state.pickup_started = False
        return False

    print("Delivery did not finish; keeping claw closed and retrying on a later frame")
    state.path_executed = True
    state.pickup_started = False
    return False


def print_debug_detections(color_matrix, vision_scene=None):
    if vision_scene is not None:
        print("Vision detections:", vision_scene.summary())

    if not ALLOW_COLOR_DETECTION_FALLBACK:
        print("Color detection fallback is disabled")
        return

    Goal_A, Goal_B = goals_pos_approx(color_matrix, "PK", "C")
    print("Color fallback Goal_A:", Goal_A)
    print("Color fallback Goal_B:", Goal_B)

    orangeball_pos = ball_pos_approx_shape(color_matrix, "O")
    print("Color fallback orangeball_pos:", orangeball_pos)


def show_frame_and_should_quit(warped_frame):
    cv.imshow("camera", warped_frame)
    return vision_live_view_quit_requested() or (cv.waitKey(1) & 0xFF) == ord("q")


def run_autonomous_camera():
    start_time = time.time()
    state = AutonomousState(
        begin_time=start_time,
        last_capture_time=start_time,
    )

    ensure_image_dir()

    camera = open_camera(CAMERA_INDEX)
    if camera is None:
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((HOST, PORT))
        if not send_command(sock, build_handshake()):
            return
        if not send_command(sock, build_mapsize(EV3_MAP_HEIGHT, EV3_MAP_WIDTH)):
            return

        camera.read()

        while camera.isOpened():
            res, frame = camera.read()

            if not res:
                continue

            warped_frame = warp_frame(frame)
            now = time.time()

            if startup_delay_has_elapsed(state, now) and capture_interval_has_elapsed(state, now):
                scene = capture_detection_scene(state, warped_frame)
                color_matrix = scene["color_matrix"]
                vision_scene = scene["vision_scene"]

                if state.path_executed:
                    if retry_delivery(sock, camera, state):
                        break
                    continue

                pickup_target = detect_pickup_target(color_matrix, vision_scene)

                if pickup_target is None:
                    continue

                if handle_pickup_and_delivery(sock, camera, state, color_matrix, pickup_target):
                    break

                print_debug_detections(color_matrix, vision_scene)

            if show_frame_and_should_quit(warped_frame):
                break

    finally:
        sock.close()
        close_camera(camera)


def main():
    run_autonomous_camera()


if __name__ == "__main__":
    main()
