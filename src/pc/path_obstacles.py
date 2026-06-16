import math

from settings import RED_CROSS_AVOIDANCE_ENABLED, RED_CROSS_OBSTACLE_MARGIN


RED_CROSS_BLOCKED_VALUE = "X"


def clone_path_matrix(color_matrix):
    return [list(row) for row in color_matrix]


def clear_path_endpoint(color_matrix, point, radius=5, value="."):
    if point is None or not color_matrix:
        return

    row_count = len(color_matrix)
    col_count = len(color_matrix[0])
    row, col = point
    row = int(round(row))
    col = int(round(col))
    radius = max(0, int(round(radius)))

    for current_row in range(max(0, row - radius), min(row_count, row + radius + 1)):
        for current_col in range(max(0, col - radius), min(col_count, col + radius + 1)):
            color_matrix[current_row][current_col] = value


def _bbox_center_matches_point(detection, tolerance=5.0):
    x1, y1, x2, y2 = detection.bbox
    center_row = (float(y1) + float(y2)) / 2.0
    center_col = (float(x1) + float(x2)) / 2.0
    point_row, point_col = detection.point

    return (
        abs(center_row - float(point_row)) <= tolerance
        and abs(center_col - float(point_col)) <= tolerance
    )


def _clamp_region(top, left, bottom, right, row_count, col_count):
    top = max(0, min(row_count - 1, int(top)))
    left = max(0, min(col_count - 1, int(left)))
    bottom = max(0, min(row_count - 1, int(bottom)))
    right = max(0, min(col_count - 1, int(right)))

    if bottom < top or right < left:
        return None

    return top, left, bottom, right


def _red_cross_region(detection, row_count, col_count, margin):
    margin = max(0, int(round(margin)))

    if _bbox_center_matches_point(detection):
        x1, y1, x2, y2 = detection.bbox
        top = math.floor(min(float(y1), float(y2))) - margin
        bottom = math.ceil(max(float(y1), float(y2))) + margin
        left = math.floor(min(float(x1), float(x2))) - margin
        right = math.ceil(max(float(x1), float(x2))) + margin
    else:
        point_row, point_col = detection.point
        top = int(round(point_row)) - margin
        bottom = int(round(point_row)) + margin
        left = int(round(point_col)) - margin
        right = int(round(point_col)) + margin

    return _clamp_region(top, left, bottom, right, row_count, col_count)


def red_cross_obstacle_regions(
    color_matrix,
    vision_scene,
    margin=RED_CROSS_OBSTACLE_MARGIN,
):
    if (
        not RED_CROSS_AVOIDANCE_ENABLED
        or color_matrix is None
        or vision_scene is None
        or not color_matrix
    ):
        return []

    row_count = len(color_matrix)
    col_count = len(color_matrix[0])
    regions = []

    for detection in vision_scene.detections_for("redcross"):
        region = _red_cross_region(detection, row_count, col_count, margin)

        if region is not None:
            regions.append(region)

    return regions


def segment_intersects_regions(start_point, end_point, regions, step_size=3.0):
    if not regions:
        return False

    start_row, start_col = start_point
    end_row, end_col = end_point
    delta_row = float(end_row) - float(start_row)
    delta_col = float(end_col) - float(start_col)
    distance = math.hypot(delta_row, delta_col)

    if distance <= 0.001:
        sample_count = 1
    else:
        sample_count = max(1, int(math.ceil(distance / max(1.0, float(step_size)))))

    for sample_index in range(sample_count + 1):
        t = float(sample_index) / float(sample_count)
        row = float(start_row) + delta_row * t
        col = float(start_col) + delta_col * t

        for top, left, bottom, right in regions:
            if top <= row <= bottom and left <= col <= right:
                return True

    return False


def mark_red_cross_obstacles(
    color_matrix,
    vision_scene,
    margin=RED_CROSS_OBSTACLE_MARGIN,
    value=RED_CROSS_BLOCKED_VALUE,
):
    regions = red_cross_obstacle_regions(color_matrix, vision_scene, margin=margin)

    for region in regions:
        top, left, bottom, right = region

        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                color_matrix[row][col] = value

    if regions:
        print(
            "Red cross avoidance: marked {} obstacle region(s) with margin {} map units".format(
                len(regions),
                int(round(margin)),
            )
        )

    return len(regions)
