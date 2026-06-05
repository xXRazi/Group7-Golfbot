import cv2 as cv
import os
import time
# from create_test_image import test_matrix
from Imagesplitter import create_matrix
from id_color import ball_pos_approx_shape, grapler_pos_approx, robot_pos, goals_pos_approx
from dotenv import load_dotenv
from collection_algorithm import A_star, get_h_list

allocatedTime = 1
STARTTIME = 2
BeginTime = time.time()
startTime = time.time()

load_dotenv()
path = os.getenv("img_path")

camera = cv.VideoCapture(0)

res, frame = camera.read()
count = 0
while camera.isOpened():
    res, frame = camera.read()

    BeginElapsedTime = time.time() - BeginTime

    if BeginElapsedTime >= STARTTIME:

        elapsedTime = time.time() - startTime

        if elapsedTime >= allocatedTime:
            elapsedTime = 0
            startTime = time.time()
            if res:
                im_ = f"{count}.png"
                full_path = os.path.join(path, im_)
                cv.imwrite(full_path, frame)
                # Directory skal være hvor du har projektet gemt
                count += 1
                print("Vi tager et billede")

                color_matrix = create_matrix(full_path)

                grapler_point = grapler_pos_approx(color_matrix, "G")

                orangeball_pos = ball_pos_approx_shape(color_matrix, "O")

                white_list = ball_pos_approx_shape(color_matrix, "W")
                min_list = []
                for item in white_list:
                    value = get_h_list(grapler_point[0],grapler_point[1],item[0],item[1])
                    min_list.append(value)
                print("minlist", min_list)
                paired = list(zip(min_list, white_list))
                paired.sort()  # sorts by min_list values
                white_list = [item for _, item in paired]

                Goal_A, Goal_B = goals_pos_approx(color_matrix, "PK", "C")

                if len(orangeball_pos) != 0:
                    robot_path = A_star(color_matrix, grapler_point, orangeball_pos)
                    for item in robot_path:
                        #goto item

                    #close grapler

                    robot_path = A_star(color_matrix, grapler_point, Goal_A)

                    for item in robot_path:
                        #goto item

                    #open grapler

                    #shoot command
                else:
                    robot_path = A_star(color_matrix, grapler_point, white_list[0])
                    for item in robot_path:
                        # goto item

                    #close grapler

                    robot_path = A_star(color_matrix, grapler_point, Goal_A)

                    #open grapler

                    #shoot

    cv.imshow("camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()