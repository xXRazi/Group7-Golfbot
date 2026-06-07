from skimage import io
import numpy as np
import cv2 as cv


def create_matrix(img_):
    img = io.imread(img_)

    # Some PNGs may contain an alpha channel. OpenCV's RGB2HSV expects 3 channels.
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    H, W, C = img.shape
    col_matrix = [["." for _ in range(W)] for _ in range(H)]

    # Order matters because the first matching label is written to col_matrix.
    # The magenta/pink robot marker overlaps the old RR/red hue range, so P must
    # be checked before RR. Otherwise the marker becomes RR and robot_pose_approx()
    # cannot find the rear marker.
    color_ranges = [
        ("W",  np.array([0, 0, 245]),     np.array([180, 18, 255])),

        # Robot marker colors.
        # Yellow is intentionally broad enough for the yellow tape under webcam light.
        ("Y",  np.array([24, 55, 75]),    np.array([42, 255, 255])),
        # Magenta/pink robot tape can be quite dark in the camera image.
        # This must come before RR.
        ("P",  np.array([145, 55, 80]),   np.array([169, 255, 255])),

        # Red arena tape. RR starts at 170 so it does not steal the P marker.
        ("R",  np.array([0, 100, 100]),   np.array([10, 255, 255])),
        ("RR", np.array([170, 100, 100]), np.array([179, 255, 255])),

        ("O",  np.array([5, 100, 120]),   np.array([22, 255, 255])),
        ("B",  np.array([0, 0, 0]),       np.array([180, 255, 55])),
        ("G",  np.array([40, 50, 50]),    np.array([80, 255, 255])),
        ("b",  np.array([85, 100, 100]),  np.array([105, 255, 255])),
        ("_",  np.array([115, 120, 120]), np.array([175, 200, 215])),
        ("PK", np.array([170, 45, 235]),  np.array([180, 80, 255])),
        ("C",  np.array([88, 45, 220]),   np.array([102, 90, 255])),
    ]

    for label, lower, upper in color_ranges:
        mask = cv.inRange(hsv_img, lower, upper)
        positions = np.where(mask > 0)

        for r, c in zip(positions[0], positions[1]):
            if col_matrix[r][c] == ".":
                col_matrix[r][c] = label

    print(f"Matrix Size: {H} rows x {W} columns\n")
    return col_matrix