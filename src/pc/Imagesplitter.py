from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv

def create_matrix(img_):
    # Load image and get dimensions
    img = io.imread(img_)
    H, W, C = img.shape
    
    # Initialize empty matrix
    col_matrix = [["." for _ in range(W)] for _ in range(H)]

    # ==========================================
    # STEP 1: PREPROCESSING (Aggressive Blur)
    # ==========================================
    blurred_img = cv.medianBlur(img, 7)
    hsv_img = cv.cvtColor(blurred_img, cv.COLOR_RGB2HSV)

    # ==========================================
    # STEP 2: ISOLATE THE ARENA
    # ==========================================
    red_mask1 = cv.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]))
    red_mask2 = cv.inRange(hsv_img, np.array([160, 100, 100]), np.array([179, 255, 255]))
    full_red_mask = cv.bitwise_or(red_mask1, red_mask2)

    contours, _ = cv.findContours(full_red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    arena_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        cv.drawContours(arena_mask, [largest_contour], -1, 255, thickness=cv.FILLED)
    else:
        arena_mask.fill(255)

    roi_hsv = cv.bitwise_and(hsv_img, hsv_img, mask=arena_mask)

    # ==========================================
    # STEP 3: PRE-SCAN FOR THE ROBOT'S LOCATION
    # ==========================================
    kernel = np.ones((5, 5), np.uint8)
    robot_found = False
    rx, ry, rw, rh = 0, 0, 0, 0

    # Temporarily look for Black ("B") to find the bounding box
    temp_black_mask = cv.inRange(roi_hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    temp_black_mask = cv.bitwise_and(temp_black_mask, temp_black_mask, mask=arena_mask)
    temp_black_clean = cv.morphologyEx(temp_black_mask, cv.MORPH_CLOSE, kernel, iterations=1)

    robot_contours, _ = cv.findContours(temp_black_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if robot_contours:
        # Find the biggest black object (the robot)
        robot_contour = max(robot_contours, key=cv.contourArea)
        rx, ry, rw, rh = cv.boundingRect(robot_contour)
        robot_found = True

    # ==========================================
    # STEP 4: COLOR DETECTION & MATRIX POPULATION
    # ==========================================
    col_dict = {
        "W": (np.array([0, 0, 245]), np.array([180, 18, 255])),
        "R": (np.array([0, 100, 100]), np.array([10, 255, 255])),
        "RR": (np.array([160, 100, 100]), np.array([179, 255, 255])),
        "O": (np.array([14, 80, 240]), np.array([30, 255, 255])),
        "B": (np.array([0, 0, 0]), np.array([180, 255, 50])),
        "G": (np.array([40, 50, 50]), np.array([80, 255, 255])), # Original green (will be overwritten if robot is found)
        "b": (np.array([85, 100, 100]), np.array([105, 255, 255])),
        "_": (np.array([115, 120, 120]), np.array([175, 200, 215])),
        "Y": (np.array([27, 85, 230]), np.array([33, 180, 255])),
        "P": (np.array([155, 130, 200]), np.array([168, 255, 255])),
        "PK": (np.array([170, 45, 235]), np.array([180, 80, 255])),
        "C": (np.array([88, 45, 220]), np.array([102, 90, 255]))
    }

    for label, (lower, upper) in col_dict.items():
        
        # --- THE GREEN CLAW OVERRIDE ---
        if label == "G" and robot_found:
            # 1. Use the wider, safer bounds for the shadowed green claw
            opt_green_lower = np.array([40, 60, 40])
            opt_green_upper = np.array([90, 255, 255])
            
            # 2. Calculate a padded box around the robot
            pad = 30
            y1, y2 = max(0, ry - pad), min(H, ry + rh + pad)
            x1, x2 = max(0, rx - pad), min(W, rx + rw + pad)
            
            # 3. Create a blank mask for the whole image
            raw_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
            
            # 4. Search for green ONLY inside the cropped robot box
            robot_crop = roi_hsv[y1:y2, x1:x2]
            green_crop_mask = cv.inRange(robot_crop, opt_green_lower, opt_green_upper)
            
            # 5. Paste the successful detections back into the full-size mask
            raw_mask[y1:y2, x1:x2] = green_crop_mask
            
        else:
            # Normal detection for all other colors
            raw_mask = cv.inRange(roi_hsv, lower, upper)

        # -------------------------------

        # Ensure no colors bleed outside the red arena walls
        constrained_mask = cv.bitwise_and(raw_mask, raw_mask, mask=arena_mask)

        # Apply morphological closing to force shiny/glare holes shut
        clean_mask = cv.morphologyEx(constrained_mask, cv.MORPH_CLOSE, kernel, iterations=1)

        # Find coordinates and populate matrix
        positions = np.where(clean_mask > 0)
        for r, c in zip(positions[0], positions[1]):
            if col_matrix[r][c] == ".":
                col_matrix[r][c] = label

    print(f"Matrix Size: {H} rows x {W} columns\n")
    return col_matrix