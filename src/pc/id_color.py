import math


# ============================================================
# Connected-component helpers
# ============================================================

def color_blobs(matrix, color, min_area=1, max_area=None):
    """Return connected blobs for one matrix color."""
    row_count = len(matrix)
    col_count = len(matrix[0])
    visited = set()
    blobs = []

    for row in range(row_count):
        for col in range(col_count):
            if matrix[row][col] != color or (row, col) in visited:
                continue

            stack = [(row, col)]
            visited.add((row, col))
            pixels = []

            while stack:
                current_row, current_col = stack.pop()
                pixels.append((current_row, current_col))

                for row_delta, col_delta in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ]:
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta

                    if 0 <= next_row < row_count and 0 <= next_col < col_count:
                        if matrix[next_row][next_col] == color and (next_row, next_col) not in visited:
                            visited.add((next_row, next_col))
                            stack.append((next_row, next_col))

            area = len(pixels)

            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            rows = [pixel[0] for pixel in pixels]
            cols = [pixel[1] for pixel in pixels]

            min_row, max_row = min(rows), max(rows)
            min_col, max_col = min(cols), max(cols)

            height = max_row - min_row + 1
            width = max_col - min_col + 1
            bbox_area = height * width

            blobs.append({
                "color": color,
                "area": area,
                "center_row": sum(rows) / area,
                "center_col": sum(cols) / area,
                "min_row": min_row,
                "max_row": max_row,
                "min_col": min_col,
                "max_col": max_col,
                "height": height,
                "width": width,
                "fill_ratio": area / bbox_area if bbox_area else 0.0,
            })

    blobs.sort(key=lambda blob: blob["area"], reverse=True)
    return blobs


def _blob_distance(blob_a, blob_b):
    return math.hypot(
        blob_a["center_row"] - blob_b["center_row"],
        blob_a["center_col"] - blob_b["center_col"],
    )


def _weighted_center(blobs):
    total_area = sum(blob["area"] for blob in blobs)

    if total_area <= 0:
        return None

    center_row = sum(blob["center_row"] * blob["area"] for blob in blobs) / total_area
    center_col = sum(blob["center_col"] * blob["area"] for blob in blobs) / total_area

    return center_row, center_col


# ============================================================
# Ball detection
# ============================================================

def _ball_shape_is_valid(blob, color):
    """
    Decide if a color blob is actually a ball.

    White false positives usually come from white clothing, robot reflections,
    white wall/goal edges, or tiny light specks. The real balls in the warped
    camera image are compact, roughly circular blobs.
    """
    area = blob["area"]
    height = blob["height"]
    width = blob["width"]
    fill_ratio = blob["fill_ratio"]

    if height <= 0 or width <= 0:
        return False

    ratio = width / height

    if color == "W":
        if not (40 <= area <= 160):
            return False
        if not (6 <= height <= 14 and 6 <= width <= 14):
            return False
        if not (0.65 <= ratio <= 1.45):
            return False
        if not (0.52 <= fill_ratio <= 0.95):
            return False
        return True

    if color == "O":
        if not (25 <= area <= 160):
            return False
        if not (5 <= height <= 16 and 5 <= width <= 16):
            return False
        if not (0.60 <= ratio <= 1.55):
            return False
        if not (0.40 <= fill_ratio <= 0.92):
            return False
        return True

    if not (25 <= area <= 160):
        return False
    if not (5 <= height <= 16 and 5 <= width <= 16):
        return False
    if not (0.60 <= ratio <= 1.55):
        return False
    if not (0.40 <= fill_ratio <= 0.92):
        return False

    return True


def ball_pos_approx_shape(matrix, color):
    """Return center points of real ball-shaped blobs as (row, col)."""
    blobs = color_blobs(matrix, color, min_area=1, max_area=2000)
    balls = []
    rejected = 0

    for blob in blobs:
        if not _ball_shape_is_valid(blob, color):
            rejected += 1
            continue

        center_row = int(round(blob["center_row"]))
        center_col = int(round(blob["center_col"]))
        balls.append((center_row, center_col))

    if rejected > 0:
        print(
            "Ball detection debug: color={}, accepted={}, rejected={}".format(
                color,
                balls,
                rejected,
            )
        )

    return balls


# ============================================================
# Grappler / goal / robot detection
# ============================================================

def grapler_pos_approx(matrix, color):
    """
    Return the grappler midpoint as (row, col).

    The old code split green pixels into exactly two groups. That failed when the
    green pieces were close together or partly merged, returning None even though
    the grappler was visible.
    """
    blobs = color_blobs(matrix, color, min_area=3, max_area=600)

    if not blobs:
        print("Grappler debug: no {} blobs found".format(color))
        return None

    largest = blobs[0]
    nearby_blobs = []

    for i in color_list:
        if abs(color_list[0][0] - i[0]) <= 10 and abs(color_list[0][1] - i[1]) <= 30:
            Grapler_one_list.append(i)
        else:
            Grapler_two_list.append(i)

    if len(Grapler_one_list) == 0 or len(Grapler_two_list) == 0:
        return None

    result = (int(round(center[0])), int(round(center[1])))

    print(
        "Grappler debug: blobs={}, used={}, result={}".format(
            len(blobs),
            len(nearby_blobs[:4]),
            result,
        )
    )

    return result


def goal_marker_pos_approx(matrix, color, side=None):
    """
    Find one goal marker robustly.

    Goal markers are chosen from connected blobs instead of averaging every pixel
    of a color.
    """
    col_count = len(matrix[0])
    blobs = color_blobs(matrix, color, min_area=3, max_area=2000)

    if not blobs:
        print("Goal marker debug: no blobs found for color {}".format(color))
        return None

    if side == "right":
        side_blobs = [blob for blob in blobs if blob["center_col"] > col_count * 0.55]

        if side_blobs:
            chosen = max(side_blobs, key=lambda blob: (blob["area"], blob["center_col"]))
        else:
            chosen = max(blobs, key=lambda blob: (blob["center_col"], blob["area"]))

    elif side == "left":
        side_blobs = [blob for blob in blobs if blob["center_col"] < col_count * 0.45]

        if side_blobs:
            chosen = max(side_blobs, key=lambda blob: (blob["area"], -blob["center_col"]))
        else:
            chosen = min(blobs, key=lambda blob: (blob["center_col"], -blob["area"]))

    else:
        chosen = max(blobs, key=lambda blob: blob["area"])

    result = (
        int(round(chosen["center_row"])),
        int(round(chosen["center_col"])),
    )

    print(
        "Goal marker debug: color={}, side={}, blobs={}, chosen={}, area={}, bbox=({}, {})".format(
            color,
            side,
            len(blobs),
            result,
            chosen["area"],
            chosen["height"],
            chosen["width"],
        )
    )

    return result


def _goal_side_hint(color, fallback=None):
    if color in ("PK", "P"):
        return "right"

    if color in ("C", "b"):
        return "left"

    return fallback


def goals_pos_approx(matrix, colorA, colorB):
    """
    Return Goal A and Goal B marker positions.

    In this project:
      Goal A = PK marker on the right side
      Goal B = C marker on the left side
    """
    side_a = _goal_side_hint(colorA, "right")
    side_b = _goal_side_hint(colorB, "left")

    goal_a = goal_marker_pos_approx(matrix, colorA, side_a)
    goal_b = goal_marker_pos_approx(matrix, colorB, side_b)

    if goal_a is None or goal_b is None:
        return None

    return goal_a, goal_b


def robot_pos(matrix):
    row_count = len(matrix)
    col_count = len(matrix[0])
    positions = []

    for row in range(row_count):
        for col in range(col_count):
            if matrix[row][col] in ("Y", "P", "B"):
                positions.append((row, col))

    return positions


def color_center(matrix, color):
    row_count = len(matrix)
    col_count = len(matrix[0])
    color_list = []

    for row in range(row_count):
        for col in range(col_count):
            if matrix[row][col] == color:
                color_list.append((row, col))

    if len(color_list) == 0:
        return None

    return (
        sum(point[0] for point in color_list) / len(color_list),
        sum(point[1] for point in color_list) / len(color_list),
    )


def _point_in_expanded_bbox(row, col, bbox_blob, margin):
    return (
        bbox_blob["min_row"] - margin <= row <= bbox_blob["max_row"] + margin and
        bbox_blob["min_col"] - margin <= col <= bbox_blob["max_col"] + margin
    )


def _distance_between_blobs(blob_a, blob_b):
    return math.hypot(
        blob_a["center_row"] - blob_b["center_row"],
        blob_a["center_col"] - blob_b["center_col"],
    )


def robot_pose_approx(matrix):
    """
    Estimate robot pose from the yellow front marker and pink rear marker.

    Returns:
      (center_col, center_row, heading_degrees)
    """
    yellow_blobs = color_blobs(matrix, "Y", min_area=8, max_area=500)
    pink_blobs = color_blobs(matrix, "P", min_area=8, max_area=500)
    black_blobs = color_blobs(matrix, "B", min_area=150)

    if not yellow_blobs or not pink_blobs:
        print(
            "Robot pose debug: missing marker blobs Y={}, P={}".format(
                len(yellow_blobs),
                len(pink_blobs),
            )
        )
        return None

    marker_margin = 55
    front_blobs = []
    rear_blobs = []

    if black_blobs:
        body = black_blobs[0]

        front_blobs = [
            blob for blob in yellow_blobs
            if _point_in_expanded_bbox(
                blob["center_row"],
                blob["center_col"],
                body,
                marker_margin,
            )
        ]

        rear_blobs = [
            blob for blob in pink_blobs
            if _point_in_expanded_bbox(
                blob["center_row"],
                blob["center_col"],
                body,
                marker_margin,
            )
        ]

    if not front_blobs or not rear_blobs:
        best_pair = None
        best_distance = None

        for yellow in yellow_blobs:
            for pink in pink_blobs:
                distance = _distance_between_blobs(yellow, pink)

                if not (15 <= distance <= 160):
                    continue

                if best_distance is None or distance < best_distance:
                    best_pair = (yellow, pink)
                    best_distance = distance

        if best_pair is None:
            print("Robot pose debug: no valid Y/P marker pair")
            return None

        front_blobs = [best_pair[0]]
        rear_blobs = [best_pair[1]]

    front = _weighted_center(front_blobs)
    rear = _weighted_center(rear_blobs)

    if front is None or rear is None:
        return None

    front_row, front_col = front
    rear_row, rear_col = rear

    center_row = (front_row + rear_row) / 2.0
    center_col = (front_col + rear_col) / 2.0

    dx = front_col - rear_col
    dy = front_row - rear_row

    if dx == 0 and dy == 0:
        print("Robot pose debug: front and rear markers overlap")
        return None

    heading = math.degrees(math.atan2(dy, dx)) % 360.0

    print(
        "Robot pose debug: body_blobs={}, Y_blobs={}, P_blobs={}, "
        "used_Y={}, used_P={}, center=({:.1f}, {:.1f}), heading={:.1f}".format(
            len(black_blobs),
            len(yellow_blobs),
            len(pink_blobs),
            len(front_blobs),
            len(rear_blobs),
            center_col,
            center_row,
            heading,
        )
    )

    return center_col, center_row, heading

def get_yolo_balls(frame, model, conf_threshold=0.5):
    """
    Runs YOLO on the current frame and returns lists of (row, col) tuples 
    for the centers of the detected balls.
    """
    results = model.predict(source=frame, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes
    names = model.names

    white_balls = []
    orange_balls = []

    for box in boxes:
        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        
        # Clean the class name just like in your test script
        class_name = names[class_id].lower().replace("_", "").replace("-", "").replace(" ", "")

        # Calculate the exact center point (row, col)
        center_row = (y1 + y2) // 2
        center_col = (x1 + x2) // 2

        if class_name == "whiteball":
            white_balls.append((center_row, center_col))
        elif class_name == "orangeball":
            orange_balls.append((center_row, center_col))

    return white_balls, orange_balls