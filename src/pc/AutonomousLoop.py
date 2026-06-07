import cv2 as cv
import os
import time
import socket
from ultralytics import YOLO # Import YOLO

from dotenv import load_dotenv
from Imagesplitter import create_matrix
from id_color import grapler_pos_approx, goals_pos_approx
# from id_color import ball_pos_approx_shape  <-- We don't need this anymore
from collection_algorithm import A_star
from com_protocol import HOST, PORT, send_command, build_handshake, build_goto, build_finish, build_open_claw, build_close_claw, build_deliver_ball, build_turn
#from scoring_and_corner import open_claw, close_claw, deliver_ball

# Paste or import the get_yolo_balls function here
def get_yolo_balls(frame, model, conf_threshold=0.5):
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    names = model.names
    white_balls, orange_balls = [], []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = names[class_id].lower().replace("_", "").replace("-", "").replace(" ", "")
        
        center_row = (y1 + y2) // 2
        center_col = (x1 + x2) // 2

        if class_name == "whiteball":
            white_balls.append((center_row, center_col))
        elif class_name == "orangeball":
            orange_balls.append((center_row, center_col))

    return white_balls, orange_balls

allocatedTime = 1
STARTTIME = 0

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(BASE_DIR, "images")
os.makedirs(path, exist_ok=True) # Create the directory if it doesn't exist

# --- LOAD YOLO MODEL ONCE HERE ---
print("Loading YOLO Model...")
model_path = "/home/sander/workspace/CDIO_live/Group7-Golfbot/runs/detect/arena_model_v18/weights/best.pt"
yolo_model = YOLO(model_path)
print("Model Loaded!")

camera = cv.VideoCapture(6)

if not camera.isOpened():
    print("Error: Could not open camera. Make sure it is connected and the correct index is used.")
    exit()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
send_command(sock, build_handshake())

def closest_point(origin, points):
    if origin is None or not points:
        return None
    return min(points, key=lambda p: ((origin[0] - p[0]) ** 2 + (origin[1] - p[1]) ** 2) ** 0.5)

def follow_path(sock, path, step_size=40):
    if not path_is_valid(path):
        print("Invalid path:", path)
        return False

    waypoints = path[::step_size]
    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])

    for row, col in waypoints:
        x, y = int(col), int(row)
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

    send_command(sock, build_open_claw())
    path_to_ball = A_star(color_matrix, grapler_point, target)

    if not follow_pickup_path_and_close(sock, path_to_ball):
        return False, count

    print("closing grapler")
    send_command(sock, build_close_claw())
    time.sleep(0.5)

    new_frame, new_color_matrix = take_picture_and_matrix(camera, count)
    count += 1

    if new_color_matrix is None:
        return False, count

    new_grapler_point = grapler_pos_approx(new_color_matrix, "G")
    new_goals = goals_pos_approx(new_color_matrix, "PK", "C")

    if new_grapler_point is None or new_goals is None:
        return False, count

    print("Driving to goal:", goal)
    path_to_goal = A_star(new_color_matrix, new_grapler_point, goal)

    if not follow_path(sock, path_to_goal):
        return False, count

    print("Shooting")
    send_command(sock, build_deliver_ball())
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
                print("No grapler detected, turning slightly and trying again.")
                send_command(sock, build_turn(10, 25))  # Turn 10 degrees at speed 25
                time.sleep(1)  # Give the robot time to turn
                continue

            print(f"Grapler detected at: {grapler_point}")

            # --- THE INTEGRATION ---
            # Pass the raw image frame to YOLO instead of the color matrix
            white_balls, orange_balls = get_yolo_balls(frame, yolo_model)
            print(f"Found {len(white_balls)} white balls and {len(orange_balls)} orange balls.")

            goals = goals_pos_approx(color_matrix, "PK", "C")
            if goals is None:
                print("Could not find goals, trying again on next frame.")
                continue
            
            print(f"Goals detected at: {goals}")

            Goal_A, Goal_B = goals

            # Priority logic remains exactly the same!
            if orange_balls:
                target = closest_point(grapler_point, orange_balls)
                goal = Goal_A
                print("Collecting orange ball:", target)
            elif white_balls:
                target = closest_point(grapler_point, white_balls)
                goal = Goal_B
                print("Collecting white ball:", target)
            else:
                print("No balls left, finishing.")
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