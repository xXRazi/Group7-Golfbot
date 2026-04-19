import cv2 as cv
import os
import time
from Imagesplitter import create_matrix
from id_color import ball_pos_approx
from dotenv import load_dotenv
from collection_algorithm import A_star

allocatedTime = 1
STARTTIME = 2
BeginTime = time.time()
startTime = time.time()

load_dotenv()
path = os.getenv("path")

camera = cv.VideoCapture(1)

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
                full_path = os.path.join(path,im_)
                cv.imwrite(full_path, frame)
                #Directory skal være hvor du har projektet gemt
                count += 1
                print("Vi tager et billede")
                color_matrix = create_matrix(full_path)

                #test to show white_lists
                white_list = ball_pos_approx(color_matrix, "W")
                print(white_list[0])
                A_star(color_matrix, white_list[0],white_list[-1])



    cv.imshow("camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
