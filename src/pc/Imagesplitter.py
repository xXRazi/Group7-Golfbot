from skimage import io
from matplotlib import pyplot as plt
from color_detection import *
import cv2 as cv

img = io.imread(im_)

arr = np.stack([img])

block_h = 16
block_w = 16

n, H, W, C = arr.shape

nH = H // block_h
nW = W // block_w

arr = arr[:, :nH*block_h, :nW*block_w, :]

col_matrix = [["." for _ in range(nW)] for _ in range(nH)]

"""

r r r r r r r r r r r r 
r _ _ _ w _ _ _ _ _ _ r
r _ _ _ _ _ _ _ _ _ _ rr
r _ _ _ _ _ _ _ _ _ _ r
r _ _ _ _ _ _ _ _ _ _ r
r r r r r r r r r r r r

r
r
r
r
r
r
Find gul bold, aflever ved nærmeste mål

connect alle hvide bolde og mål
Gå til tætteste hvide bold, gå til tætteste mål(udfra calc med mål A eller mål B)

Side note: Hvordan finder vi et mål?


R 

col

R 

"""

col_dict = {
    "W": (np.array([0, 0, 200]), np.array([180, 50, 255])),
    "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
    "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
    "O": (np.array([10, 100, 100]), np.array([25, 255, 255])),
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



    plt.subplot(nH, nW, i + 1)
    plt.imshow(block)
    plt.axis("off")
    plt.title(f"{i}", fontsize=6)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
