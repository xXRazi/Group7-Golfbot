import cv2 as cv
import os
import time
import socket

from dotenv import load_dotenv
from Imagesplitter import create_matrix
from id_color import ball_pos_approx_shape, grapler_pos_approx, goals_pos_approx
from collection_algorithm import A_star
from com_protocol import HOST, PORT, send_command, build_handshake, build_goto, build_finish
from scoring_and_corner import open_claw, close_claw, deliver_ball

allocatedTime = 1
STARTTIME = 2

load_dotenv()
path = os.getenv("img_path")

camera = cv.VideoCapture(0)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
send_command(sock, build_handshake())


def closest_point(origin, points):
    if origin is None or not points:
        return None

    return min(
        points,
        key=lambda p: ((origin[0] - p[0]) ** 2 + (origin[1] - p[1]) ** 2) ** 0.5
    )


def follow_path(sock, path, step_size=40):
    if not path or isinstance(path, str):
        print("Invalid path:", path)
        return False

    waypoints = path[::step_size]

    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])

    for row, col in waypoints:
        x = int(col)
        y = int(row)

        print("Goto:", x, y)

        if not send_command(sock, build_goto(x, y)):
            return False

        time.sleep(0.2)

    return True


def take_picture_and_matrix(camera, count):
    res, frame = camera.read()

    if not res:
        return None, None

    im_ = f"{count}.png"
    full_path = os.path.join(path, im_)
    cv.imwrite(full_path, frame)

    color_matrix = create_matrix(full_path)

    return frame, color_matrix


def collect_ball(color_matrix, sock, camera, count, target, goal):
    grapler_point = grapler_pos_approx(color_matrix, "G")

    if grapler_point is None:
        print("Could not find grapler")
        return False, count

    open_claw()

    path_to_ball = A_star(color_matrix, grapler_point, target)

    if not follow_path(sock, path_to_ball):
        return False, count

    print("closing grapler")
    close_claw()

    time.sleep(0.5)  # give robot/claw time to settle

    # Take new image after pickup
    new_frame, new_color_matrix = take_picture_and_matrix(camera, count)
    count += 1

    if new_color_matrix is None:
        print("Could not take new image after pickup")
        return False, count

    new_grapler_point = grapler_pos_approx(new_color_matrix, "G")

    if new_grapler_point is None:
        print("Could not re-detect grapler after pickup")
        return False, count

    # Re-detect goal too, in case camera/image changed
    new_goals = goals_pos_approx(new_color_matrix, "PK", "C")

    if new_goals is None:
        print("Could not re-detect goals after pickup")
        return False, count

    Goal_A, Goal_B = new_goals

    # Keep using the same chosen goal type
    updated_goal = goal

    print("New grapler position:", new_grapler_point)
    print("Driving from new grapler position to goal:", updated_goal)

    path_to_goal = A_star(new_color_matrix, new_grapler_point, updated_goal)

    if not follow_path(sock, path_to_goal):
        return False, count

    print("Shooting")
    deliver_ball()

    return True, count


count = 0
BeginTime = time.time()
startTime = time.time()

try:
    while camera.isOpened():
        res, preview_frame = camera.read()

        if res:
            cv.imshow("camera", preview_frame)

        if time.time() - BeginTime < STARTTIME:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        if time.time() - startTime >= allocatedTime:
            startTime = time.time()

            frame, color_matrix = take_picture_and_matrix(camera, count)
            count += 1

            if color_matrix is None:
                continue

            print("Image taken")

            grapler_point = grapler_pos_approx(color_matrix, "G")

            if grapler_point is None:
                print("No grapler detected")
                continue

            white_balls = ball_pos_approx_shape(color_matrix, "W")
            orange_balls = ball_pos_approx_shape(color_matrix, "O")

            goals = goals_pos_approx(color_matrix, "PK", "C")

            if goals is None:
                print("Could not find goals")
                continue

            Goal_A, Goal_B = goals

            # Priority: collect orange first if visible
            if orange_balls:
                target = closest_point(grapler_point, orange_balls)
                goal = Goal_A
                print("Collecting orange ball:", target)

            elif white_balls:
                target = closest_point(grapler_point, white_balls)
                goal = Goal_B
                print("Collecting white ball:", target)

            else:
                print("No balls left")
                send_command(sock, build_finish())
                break

            success, count = collect_ball(color_matrix, sock, camera, count, target, goal)

            if not success:
                print("Collection failed, trying again on next frame")

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    sock.close()
    camera.release()
    cv.destroyAllWindows()