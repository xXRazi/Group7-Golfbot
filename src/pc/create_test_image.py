from Imagesplitter import create_matrix
from dotenv import load_dotenv
import os
import cv2
import numpy as np

load_dotenv()
img_path = os.getenv("img_path")

test_path = img_path + "/1.png"

test_matrix = create_matrix(test_path)

"""frame = cv2.imread("C:/Users/ronik/Desktop/ComputerEngineering/4_Semester/CDIO_project/Img_CDIO/1.png")  # or your camera frame
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = hsv[y, x]
        print(f"x={x}, y={y}  HSV=({h}, {s}, {v})")

cv2.imshow("frame", frame)
cv2.setMouseCallback("frame", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()"""