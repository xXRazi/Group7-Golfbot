import math


def ball_pos_approx_shape(matrix, color):
    row = len(matrix)
    col = len(matrix[0])

    visited = set()
    balls = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] != color or (i, j) in visited:
                continue

            stack = [(i, j)]
            blob = []
            visited.add((i, j))

            while stack:
                r, c = stack.pop()
                blob.append((r, c))

                for dr, dc in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)
                ]:
                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < row and 0 <= nc < col:
                        if matrix[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

            area = len(blob)

            rows = [p[0] for p in blob]
            cols = [p[1] for p in blob]

            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)

            height = max_r - min_r + 1
            width = max_c - min_c + 1

            if height == 0 or width == 0:
                continue

            ratio = width / height
            bbox_area = width * height
            fill_ratio = area / bbox_area

            if color == "O":
                if not (20 < area < 180):
                    continue
                if not (0.75 < ratio < 1.25):
                    continue
                if not (0.45 < fill_ratio < 0.90):
                    continue

            elif color == "W":
                if not (15 < area < 350):
                    continue
                if not (0.45 < ratio < 1.8):
                    continue
                if not (0.25 < fill_ratio < 0.95):
                    continue

            center_r = int(sum(rows) / area)
            center_c = int(sum(cols) / area)

            balls.append((center_r, center_c))

    return balls


def grapler_pos_approx(matrix, color):
    row = len(matrix)
    col = len(matrix[0])

    color_list = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == color:
                color_list.append((i, j))

    if len(color_list) == 0:
        return None

    Grapler_one_list = []
    Grapler_two_list = []

    for i in color_list:
        if abs(color_list[0][0] - i[0]) <= 10 and abs(color_list[0][1] - i[1]) <= 10:
            Grapler_one_list.append(i)
        else:
            Grapler_two_list.append(i)

    if len(Grapler_one_list) == 0 or len(Grapler_two_list) == 0:
        return None

    Grapler_one_average = (
        sum(p[0] for p in Grapler_one_list) // len(Grapler_one_list),
        sum(p[1] for p in Grapler_one_list) // len(Grapler_one_list)
    )

    Grapler_two_average = (
        sum(p[0] for p in Grapler_two_list) // len(Grapler_two_list),
        sum(p[1] for p in Grapler_two_list) // len(Grapler_two_list)
    )

    Grapler_midpoint = (
        (Grapler_one_average[0] + Grapler_two_average[0]) // 2,
        (Grapler_one_average[1] + Grapler_two_average[1]) // 2
    )

    return Grapler_midpoint


def color_blobs(matrix, color, min_area=1, max_area=None):
    """Return connected blobs for one matrix color."""
    row = len(matrix)
    col = len(matrix[0])
    visited = set()
    blobs = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] != color or (i, j) in visited:
                continue

            stack = [(i, j)]
            visited.add((i, j))
            pixels = []

            while stack:
                r, c = stack.pop()
                pixels.append((r, c))

                for dr, dc in [
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)
                ]:
                    nr = r + dr
                    nc = c + dc

                    if 0 <= nr < row and 0 <= nc < col:
                        if matrix[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

            area = len(pixels)

            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            rows = [p[0] for p in pixels]
            cols = [p[1] for p in pixels]

            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)

            height = max_r - min_r + 1
            width = max_c - min_c + 1
            bbox_area = height * width

            blobs.append({
                "color": color,
                "area": area,
                "center_row": sum(rows) / area,
                "center_col": sum(cols) / area,
                "min_row": min_r,
                "max_row": max_r,
                "min_col": min_c,
                "max_col": max_c,
                "height": height,
                "width": width,
                "fill_ratio": area / bbox_area if bbox_area else 0.0,
            })

    blobs.sort(key=lambda blob: blob["area"], reverse=True)
    return blobs


def goal_marker_pos_approx(matrix, color, side=None):
    """
    Find one goal marker robustly.

    The old goals_pos_approx() averaged all pixels of a color. That could make
    the robot think the goal was near the robot if some robot tape or noise had
    the same color.

    This version uses connected blobs and chooses:
      - rightmost/largest marker for the right-side PK goal
      - leftmost/largest marker for the left-side C goal
    """
    row_count = len(matrix)
    col_count = len(matrix[0])

    blobs = color_blobs(matrix, color, min_area=3, max_area=2000)

    if not blobs:
        print("Goal marker debug: no blobs found for color {}".format(color))
        return None

    if side == "right":
        side_blobs = [
            blob for blob in blobs
            if blob["center_col"] > col_count * 0.55
        ]

        if side_blobs:
            chosen = max(side_blobs, key=lambda blob: (blob["area"], blob["center_col"]))
        else:
            chosen = max(blobs, key=lambda blob: (blob["center_col"], blob["area"]))

    elif side == "left":
        side_blobs = [
            blob for blob in blobs
            if blob["center_col"] < col_count * 0.45
        ]

        if side_blobs:
            chosen = max(side_blobs, key=lambda blob: (blob["area"], -blob["center_col"]))
        else:
            chosen = min(blobs, key=lambda blob: (blob["center_col"], -blob["area"]))

    else:
        chosen = max(blobs, key=lambda blob: blob["area"])

    result = (
        int(round(chosen["center_row"])),
        int(round(chosen["center_col"]))
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

    The markers are now inside the goals. These returned positions are therefore
    aiming markers, not drive-to positions.
    """
    side_a = _goal_side_hint(colorA, "right")
    side_b = _goal_side_hint(colorB, "left")

    goal_a = goal_marker_pos_approx(matrix, colorA, side_a)
    goal_b = goal_marker_pos_approx(matrix, colorB, side_b)

    if goal_a is None or goal_b is None:
        return None

    return goal_a, goal_b


def robot_pos(matrix):
    row = len(matrix)
    col = len(matrix[0])

    robot_pos = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == "Y":
                robot_pos.append((i, j))
            if matrix[i][j] == "P":
                robot_pos.append((i, j))
            if matrix[i][j] == "B":
                robot_pos.append((i, j))

    return robot_pos


def color_center(matrix, color):
    row = len(matrix)
    col = len(matrix[0])

    color_list = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == color:
                color_list.append((i, j))

    if len(color_list) == 0:
        return None

    return (
        sum(p[0] for p in color_list) / len(color_list),
        sum(p[1] for p in color_list) / len(color_list)
    )


def _point_in_expanded_bbox(row, col, bbox_blob, margin):
    return (
        bbox_blob["min_row"] - margin <= row <= bbox_blob["max_row"] + margin and
        bbox_blob["min_col"] - margin <= col <= bbox_blob["max_col"] + margin
    )


def _weighted_center(blobs):
    total_area = sum(blob["area"] for blob in blobs)

    if total_area <= 0:
        return None

    row = sum(blob["center_row"] * blob["area"] for blob in blobs) / total_area
    col = sum(blob["center_col"] * blob["area"] for blob in blobs) / total_area

    return row, col


def _distance_between_blobs(a, b):
    return math.hypot(
        a["center_row"] - b["center_row"],
        a["center_col"] - b["center_col"]
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
        print("Robot pose debug: missing marker blobs Y={}, P={}".format(
            len(yellow_blobs),
            len(pink_blobs)
        ))
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
                marker_margin
            )
        ]

        rear_blobs = [
            blob for blob in pink_blobs
            if _point_in_expanded_bbox(
                blob["center_row"],
                blob["center_col"],
                body,
                marker_margin
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