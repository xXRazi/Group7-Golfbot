from skimage import io
from matplotlib import pyplot as plt
from color_detection import *
import cv2 as cv

def create_matrix(img_):
    img = io.imread(img_)

    arr = np.stack([img])

    block_h = 16
    block_w = 16

    n, H, W, C = arr.shape

    nH = H // block_h
    nW = W // block_w

    arr = arr[:, :nH*block_h, :nW*block_w, :]

    col_matrix = [["." for _ in range(nW)] for _ in range(nH)]

    col_dict = {
        "W": (np.array([0, 0, 225]), np.array([180, 30, 255])),
        "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
        "O": (np.array([25, 100, 100]), np.array([25, 255, 255])),
        "B": (np.array([0,0,0]), np.array([180,255,50])),
        "G": (np.array([40,50,50]), np.array([80,255,255])),
        "_": (np.array([115,120,120]), np.array([175,200,215]))
    }

    arr_blocks = arr.reshape(
        n,
        nH, block_h,
        nW, block_w,
        C
    ).swapaxes(2, 3)

    arr_blocks = arr_blocks.reshape(-1, block_h, block_w, C)

    plt.figure(figsize=(12, 12))

    for i, block in enumerate(arr_blocks):

        row = i // nW
        col = i % nW

        hsv_block = cv.cvtColor(block, cv.COLOR_RGB2HSV)

        for label, (lower, upper) in col_dict.items():
            mask = cv.inRange(hsv_block, lower, upper)
            if cv.countNonZero(mask) > 0:
                col_matrix[row][col] = label
                break

    #from image_timer import *

        # --- 4. Print the Resulting Matrix ---
    print(f"Matrix Size: {nH} rows x {nW} columns\n")
    for row in col_matrix:
        print(" ".join(row))
    return col_matrix

