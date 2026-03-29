import cv2 as cv
import os
import time

allocatedTime = 1
startTime = time.time()

camera = cv.VideoCapture(1)

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
            cv.imwrite(os.path.join('/home/sander/Downloads', im_), frame) 
            #Directory skal være hvor du har projektet gemt
            count += 1
            print("Vi tager et billede")

    cv.imshow("camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
