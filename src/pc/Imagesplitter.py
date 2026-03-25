from skimage import io
from matplotlib import pyplot as plt
import numpy as np

img = io.imread("2.png")

arr = np.stack([img])

block_h = 16
block_w = 16

n, H, W, C = arr.shape

nH = H // block_h
nW = W // block_w

arr = arr[:, :nH*block_h, :nW*block_w, :]

arr_blocks = arr.reshape(
    n,
    nH, block_h,
    nW, block_w,
    C
).swapaxes(2, 3)

arr_blocks = arr_blocks.reshape(-1, block_h, block_w, C)

plt.figure(figsize=(12, 12))

for i, block in enumerate(arr_blocks):

    plt.subplot(nH, nW, i + 1)
    plt.imshow(block)
    plt.axis("off")
    plt.title(f"{i}", fontsize=6)

plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()
