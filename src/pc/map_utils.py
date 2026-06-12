import math

from settings import MAP_HEIGHT, MAP_WIDTH, PICKUP_STOP_DISTANCE


def path_is_valid(robot_path):
    return robot_path and not isinstance(robot_path, str)


def point_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def map_point_is_valid(point):
    row, col = point
    return 0 <= row < MAP_HEIGHT and 0 <= col < MAP_WIDTH


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_map_point(point, margin=2):
    row, col = point
    return (
        int(round(clamp(row, margin, MAP_HEIGHT - 1 - margin))),
        int(round(clamp(col, margin, MAP_WIDTH - 1 - margin))),
    )


def heading_from_map_points(start, end):
    """Return image/map heading from start(row, col) to end(row, col)."""
    start_row, start_col = start
    end_row, end_col = end
    delta_row = float(end_row - start_row)
    delta_col = float(end_col - start_col)

    if delta_row == 0.0 and delta_col == 0.0:
        return 0.0

    return math.degrees(math.atan2(delta_row, delta_col)) % 360.0


def center_point_for_grappler_point(grappler_point, heading, forward_offset, lateral_offset):
    """Convert desired grappler marker point into desired robot-center point."""
    grappler_row, grappler_col = grappler_point
    heading_rad = math.radians(heading)
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)

    center_col = (
        float(grappler_col)
        - forward_offset * cos_h
        + lateral_offset * sin_h
    )
    center_row = (
        float(grappler_row)
        - forward_offset * sin_h
        - lateral_offset * cos_h
    )

    return int(round(center_row)), int(round(center_col))


def robot_center_point(robot_pose):
    center_x, center_y, _heading = robot_pose
    return int(round(center_y)), int(round(center_x))


def truncate_path_before_target(robot_path, stop_distance=PICKUP_STOP_DISTANCE):
    """
    Return a path that stops before the final target.

    The final A* point is normally the ball center. If the robot drives all the
    way to that point, the claw/robot can hit the ball and push it away.
    """
    if not path_is_valid(robot_path):
        print("Invalid path:", robot_path)
        return None

    if len(robot_path) < 2:
        return robot_path

    distance_from_target = 0.0

    for index in range(len(robot_path) - 1, 0, -1):
        distance_from_target += point_distance(robot_path[index], robot_path[index - 1])

        if distance_from_target >= stop_distance:
            pickup_path = robot_path[:index]

            if len(pickup_path) == 0:
                pickup_path = [robot_path[0]]

            print(
                "Pickup approach: ball={}, stop_point={}, stop_distance={:.1f}".format(
                    robot_path[-1],
                    pickup_path[-1],
                    distance_from_target,
                )
            )
            return pickup_path

    print(
        "Pickup approach: path shorter than stop distance; staying at start {}".format(
            robot_path[0]
        )
    )
    return [robot_path[0]]


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def simplify_path_for_robot(robot_path, min_spacing=40):
    """Reduce one-pixel A* paths to a few stable GOTO waypoints."""
    if not path_is_valid(robot_path):
        return robot_path

    if len(robot_path) <= 2:
        return robot_path

    min_spacing = max(1.0, float(min_spacing))
    waypoints = [robot_path[0]]
    previous_direction = None

    for index in range(1, len(robot_path)):
        previous_point = robot_path[index - 1]
        current_point = robot_path[index]
        current_direction = (
            sign(current_point[0] - previous_point[0]),
            sign(current_point[1] - previous_point[1]),
        )

        direction_changed = (
            previous_direction is not None
            and current_direction != previous_direction
        )
        far_enough = point_distance(waypoints[-1], current_point) >= min_spacing

        if direction_changed and point_distance(waypoints[-1], previous_point) >= 8.0:
            if waypoints[-1] != previous_point:
                waypoints.append(previous_point)
        elif far_enough:
            waypoints.append(current_point)

        previous_direction = current_direction

    if waypoints[-1] != robot_path[-1]:
        waypoints.append(robot_path[-1])

    cleaned = [waypoints[0]]
    for point in waypoints[1:]:
        if point == robot_path[-1] or point_distance(cleaned[-1], point) >= 6.0:
            cleaned.append(point)

    print(
        "Simplified path: raw_points={}, waypoints={}".format(
            len(robot_path),
            len(cleaned),
        )
    )
    return cleaned
