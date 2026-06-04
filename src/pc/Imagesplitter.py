from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def create_matrix(img_):
    img = io.imread(img_)

    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    H, W, C = img.shape

    col_matrix = [["." for _ in range(W)] for _ in range(H)]

    col_dict = {
        "W": (np.array([0, 0, 245]), np.array([180, 18, 255])),
        "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
        "O": (np.array([25, 100, 100]), np.array([25, 255, 255])),
        "B": (np.array([0, 0, 0]), np.array([180, 255, 50])),
        "G": (np.array([40, 50, 50]), np.array([80, 255, 255])),
        "P": (np.array([160, 50, 150]), np.array([180,150,255])),
        "b": (np.array([85,100,100]), np.array([105,255,255])),
        "_": (np.array([115, 120, 120]), np.array([175, 200, 215])),
        "Y": (np.array([20, 80, 180]), np.array([35, 255, 255])),
        "P": (np.array([155, 80, 120]), np.array([175, 255, 255]))
    }

    for label, (lower, upper) in col_dict.items():
        mask = cv.inRange(hsv_img, lower, upper)

        positions = np.where(mask > 0)

        for r, c in zip(positions[0], positions[1]):
            if col_matrix[r][c] == ".":
                col_matrix[r][c] = label

    print(f"Matrix Size: {H} rows x {W} columns\n")
    return col_matrix
#hej
