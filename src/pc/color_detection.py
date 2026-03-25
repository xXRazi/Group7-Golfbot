import cv2
import cv2 as cv
import numpy as np
import os
import time

from PIL import Image

orange = [0,165,255]
red = [0,0,255]
white = [0,0,0]
yellow = [0,255,255]

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
            cv.imwrite(os.path.join('/home/sander/Downloads', f"{count}.png"), frame)
            count += 1
            print("VI tager et billede")

    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    white_lower = np.array([0,0,200],dtype=np.uint8)
    white_upper = np.array([180,30,255],dtype=np.uint8)
    white_mask = cv.inRange(hsvFrame,white_lower,white_upper)

    red_lower1 = np.array([0,100,100],dtype=np.uint8)
    red_upper1 = np.array([10,255,255],dtype=np.uint8)
    red_mask1 = cv.inRange(hsvFrame,red_lower1,red_upper1)

    red_lower2 = np.array([160,100,100],dtype=np.uint8)
    red_upper2 = np.array([180,255,255],dtype=np.uint8)
    red_mask2 = cv.inRange(hsvFrame, red_lower2, red_upper2)

    orange_lower = np.array([10,100,100],dtype=np.uint8)
    orange_upper = np.array([25,255,255],dtype=np.uint8)
    orange_mask = cv.inRange(hsvFrame,orange_lower,orange_upper)

    kernal = np.ones((5,5),"uint8")

    red_mask1 = cv.dilate(red_mask1, kernal)
    res_red1 = cv.bitwise_and(hsvFrame,hsvFrame,mask=red_mask1)

    red_mask2 = cv.dilate(red_mask2, kernal)
    res_red2 = cv.bitwise_and(hsvFrame,hsvFrame,mask=red_mask2)

    white_mask = cv.dilate(white_mask, kernal)
    res_white = cv.bitwise_and(hsvFrame,hsvFrame,mask=white_mask)

    orange_mask = cv.dilate(orange_mask, kernal)
    res_orange = cv.bitwise_and(hsvFrame,hsvFrame,mask=orange_mask)

    #contours, hierarchy = cv.findContours(red_mask1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #for pic, contour in enumerate(contours):
        #area = cv.contourArea(contour)
       # if (area > 300):
           # x, y, w, h = cv2.boundingRect(contour)
          #  imageFrame = cv2.rectangle(frame, (x, y),
         #                              (x + w, y + h),
        #                               (0, 0, 255), 2)

     #       cv2.putText(frame, "Red Colour", (x, y),
      #                  cv.FONT_HERSHEY_SIMPLEX, 1.0,
       #                 (0, 0, 255))

    #contours, hierarchy = cv.findContours(red_mask2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #for pic, contour in enumerate(contours):
     #   area = cv.contourArea(contour)
      #  if (area > 300):
       #     x, y, w, h = cv2.boundingRect(contour)
        #    imageFrame = cv2.rectangle(frame, (x, y),
         #                              (x + w, y + h),
          #                             (0, 0, 255), 2)

           # cv2.putText(frame, "Red Colour", (x, y),
            #            cv.FONT_HERSHEY_SIMPLEX, 1.0,
             #           (0, 0, 255))

    contours, hierarchy = cv.findContours(white_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(frame, "White Colour", (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    contours, hierarchy = cv.findContours(orange_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(frame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(frame, "Orange Colour", (x, y),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))


    cv.imshow("camera", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()

cv.destroyAllWindows()
