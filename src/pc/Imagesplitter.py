from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def create_matrix(img_):
    # skimage loads images in RGB format
    img = io.imread(img_)
    H, W, C = img.shape
    
    # Initialize your empty matrix
    col_matrix = [["." for _ in range(W)] for _ in range(H)]

    # ==========================================
    # STEP 1: PREPROCESSING (Aggressive Blur)
    # ==========================================
    # Apply Median Blur to the original image to smooth out sharp glare
    # on the plastic/tape before converting to HSV.
    blurred_img = cv.medianBlur(img, 7)

    # Convert to HSV (using RGB2HSV because of skimage)
    hsv_img = cv.cvtColor(blurred_img, cv.COLOR_RGB2HSV)

    # ==========================================
    # STEP 2: ISOLATE THE ARENA
    # ==========================================
    # Find the red arena to mask out the massive floor glare outside of it.
    # We use standard red wrap-around bounds here.
    red_mask1 = cv.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv.inRange(hsv_img, np.array([160, 100, 100]), np.array([179, 255, 255]))
    full_red_mask = cv.bitwise_or(red_mask1, red_mask2)

    # Find contours of the red mask
    contours, _ = cv.findContours(full_red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Create a blank canvas to draw our arena mask
    arena_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    if contours:
        # Find the largest contour (the arena walls) and fill it in solidly (255)
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(arena_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    else:
        # Fallback: If no red arena is found, just use the whole image
        arena_mask.fill(255)

    # Black out everything outside the arena
    roi_hsv = cv.bitwise_and(hsv_img, hsv_img, mask=arena_mask)

    # ==========================================
    # STEP 3: COLOR DETECTION & MATRIX POPULATION
    # ==========================================
    col_dict = {
        "W": (np.array([0, 0, 245]), np.array([180, 18, 255])),
        "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
        "O": (np.array([14, 80, 240]), np.array([30, 255, 255])),
        "B": (np.array([0, 0, 0]), np.array([180, 255, 50])),
        "G": (np.array([40, 60, 40]), np.array([90, 255, 255])),
        "b": (np.array([85, 100, 100]), np.array([105, 255, 255])),
        "_": (np.array([115, 120, 120]), np.array([175, 200, 215])),
        "Y": (np.array([27, 85, 230]), np.array([33, 180, 255])),
        "P": (np.array([155, 130, 200]), np.array([168, 255, 255])),
        "PK": (np.array([170, 45, 235]), np.array([180, 80, 255])),
        "C": (np.array([88, 45, 220]), np.array([102, 90, 255]))
    }

    # Kernel for morphological closing to patch holes in shiny objects
    kernel = np.ones((5, 5), np.uint8)

    for label, (lower, upper) in col_dict.items():
        # Check colors strictly inside the Region of Interest (roi_hsv)
        raw_mask = cv.inRange(roi_hsv, lower, upper)

        # IMPORTANT: Erase anything detected outside the arena.
        # This prevents the blacked-out exterior from being labeled as "B" (Black).
        constrained_mask = cv.bitwise_and(raw_mask, raw_mask, mask=arena_mask)

        # Apply morphological closing to force shiny/glare holes shut
        clean_mask = cv.morphologyEx(constrained_mask, cv.MORPH_CLOSE, kernel, iterations=1)

        # Find all coordinates where the color is detected
        positions = np.where(clean_mask > 0)

        # Populate the matrix
        for r, c in zip(positions[0], positions[1]):
            if col_matrix[r][c] == ".":
                # Note: "RR" will output as "RR" in the matrix. 
                # If you want it to just output "R", change the line below to:
                # col_matrix[r][c] = "R" if label == "RR" else label
                col_matrix[r][c] = label

    print(f"Matrix Size: {H} rows x {W} columns\n")
    return col_matrix

# Example of how you would call it:
# matrix = create_matrix('arena.jpg')