import cv2 as cv
import os
import time
#from create_test_image import test_matrix
from Imagesplitter import create_matrix
from id_color import ball_pos_approx, grapler_pos_approx, robot_pos, goals_pos_approx
from dotenv import load_dotenv
from collection_algorithm import A_star, get_h_list
import numpy as np


# ==========================================
# 1. PERSPECTIVE WARP SETUP
# ==========================================

# Define your desired final grid size (640 columns by 480 rows)
width, height = 640, 480 

# --- pts1: The Raw Camera Corners ---
# You still MUST measure these from your raw camera feed! 
# I am using placeholder numbers here. If your physical arena isn't 
# a perfect rectangle in the camera's eye, these numbers will not form a perfect box.
pts1 = np.float32([
    [1, 0],   # Top-Left 
    [636, 1],   # Top-Right 
    [639, 476],   # Bottom-Left 
    [1, 478]   # Bottom-Right 
]) 

# --- pts2: The Flat 2D Destination Grid ---
# This forces whatever is inside pts1 to stretch and pin to the exact 
# corners of a perfect 640x480 mathematical grid.
pts2 = np.float32([
    [0, 0],             # Top-Left pinned to 0,0
    [width, 0],         # Top-Right pinned to 640,0
    [0, height],        # Bottom-Left pinned to 0,480
    [width, height]     # Bottom-Right pinned to 640,480
])

# Compute the transformation matrix
warp_matrix = cv.getPerspectiveTransform(pts1, pts2)
# ==========================================

allocatedTime = 1
STARTTIME = 2
BeginTime = time.time()
startTime = time.time()

load_dotenv()
path = os.getenv("img_path")

camera = cv.VideoCapture(1)

res, frame = camera.read()
count = 0
while camera.isOpened():
    res, frame = camera.read()

    warped_frame = cv.warpPerspective(frame, warp_matrix, (width, height))

    BeginElapsedTime = time.time() - BeginTime

    if BeginElapsedTime >= STARTTIME:

        elapsedTime = time.time() - startTime

        if elapsedTime >= allocatedTime:
            elapsedTime = 0
            startTime = time.time()
            if res:
                im_ = f"{count}.png"
                full_path = os.path.join(path,im_)
                cv.imwrite(full_path, warped_frame)
                #Directory skal være hvor du har projektet gemt
                count += 1
                print("Vi tager et billede")
                #color_matrix = create_matrix(full_path)

                t = time.time()
                color_matrix = create_matrix(full_path)
                print("create_matrix:", time.time() - t)

                t = time.time()
                white_list = ball_pos_approx(color_matrix, "W")
                print("ball_pos:", time.time() - t)

                
                t = time.time()
                grapler_point = grapler_pos_approx(color_matrix, "G")
                print(grapler_point)
                print("grapler:", time.time() - t)

                t = time.time()
                min_list = []
                for item in white_list:
                    value = get_h_list(grapler_point[0],grapler_point[1],item[0],item[1])
                    min_list.append(value)
                print("minlist", min_list)
                paired = list(zip(min_list, white_list))
                paired.sort()  # sorts by min_list values
                white_list = [item for _, item in paired]
                print("white_list", white_list)
                robot_path = A_star(color_matrix, grapler_point, white_list[0])
                #robot_path = A_star(color_matrix, white_list[0], white_list[-1])
                print("A_star:", time.time() - t)

                robot_position= robot_pos(color_matrix)
                #print("robot_position", robot_position)

                Goal_A, Goal_B = goals_pos_approx(color_matrix, "PK", "C")
                print("Goal_A:", Goal_A)
                print("Goal_B:", Goal_B)

    cv.imshow("camera", warped_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
