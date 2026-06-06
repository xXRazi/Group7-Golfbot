from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def create_matrix(img_):
    img = io.imread(img_)

    # A. Median Blur to remove sharp glare spots
    #blurred = cv.medianBlur(img, 5) 
    
    # B. Uneven Illumination Correction (CLAHE)
    # Convert to LAB to isolate the Lightness channel
    #lab = cv.cvtColor(blurred, cv.COLOR_BGR2LAB)
    #l_channel, a_channel, b_channel = cv.split(lab)
    
    # Apply CLAHE to the L channel
    #clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #cl = clahe.apply(l_channel)
    
    # Merge back and convert to BGR, then to HSV
    #merged_lab = cv.merge((cl, a_channel, b_channel))
    #balanced_bgr = cv.cvtColor(merged_lab, cv.COLOR_LAB2BGR)
    #hsv_img = cv.cvtColor(balanced_bgr, cv.COLOR_BGR2HSV)

    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    H, W, C = img.shape

    col_matrix = [["." for _ in range(W)] for _ in range(H)]

    col_dict = {
        "W": (np.array([0, 0, 245]), np.array([180, 18, 255])),
        "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
        "O": (np.array([14, 80, 240]), np.array([30, 255, 255])),
        "B": (np.array([0, 0, 0]), np.array([180, 255, 50])),
        "G": (np.array([40, 50, 50]), np.array([80, 255, 255])),
        "b": (np.array([85,100,100]), np.array([105,255,255])),
        "_": (np.array([115, 120, 120]), np.array([175, 200, 215])),
        "Y": (np.array([27, 85, 230]), np.array([33, 180, 255])),
        "P": (np.array([155, 130, 200]), np.array([168, 255, 255])),
        "PK": (np.array([170, 45, 235]), np.array([180, 80, 255])),
        "C": (np.array([88, 45, 220]), np.array([102, 90, 255]))
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
