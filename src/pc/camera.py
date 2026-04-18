import cv2 as cv
import os
import time
from ImageSplitter import create_matrix

allocatedTime = 1
startTime = time.time()
path = "DIN PATH HER, BOZO"

camera = cv.VideoCapture(0)

res, frame = camera.read()
count = 0
while camera.isOpened():
    res, frame = camera.read()

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

    cv.imshow("camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
