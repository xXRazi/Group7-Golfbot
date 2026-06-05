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

                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1),
                               (-1,-1), (-1,1), (1,-1), (1,1)]:
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
                if not (20 < area < 250):
                    continue
                if not (0.7 < ratio < 1.3):
                    continue
                if not (0.45 < fill_ratio < 0.85):
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


def goals_pos_approx(matrix, colorA, colorB):
    row = len(matrix)
    col = len(matrix[0])
    color_list_A = []
    color_list_B = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == colorA:
                color_list_A.append((i, j))
            if matrix[i][j] == colorB:
                color_list_B.append((i, j))

    if len(color_list_A) == 0 or len(color_list_B) == 0:
        return None

    GoalA_pos = (
        sum(p[0] for p in color_list_A) // len(color_list_A),
        sum(p[1] for p in color_list_A) // len(color_list_A)
    )

    GoalB_pos = (
        sum(p[0] for p in color_list_B) // len(color_list_B),
        sum(p[1] for p in color_list_B) // len(color_list_B)
    )

    return GoalA_pos, GoalB_pos
    
    """
    Husk når du assigner skal du gøre det inorder 
    A, B = goals_pos_approx()
    """

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


def robot_pose_approx(matrix):
    front = color_center(matrix, "Y")
    rear = color_center(matrix, "P")

    if front is None or rear is None:
        return None

    front_row, front_col = front
    rear_row, rear_col = rear

    center_row = (front_row + rear_row) / 2
    center_col = (front_col + rear_col) / 2

    dx = front_col - rear_col
    dy = front_row - rear_row

    heading = math.degrees(math.atan2(dy, dx)) % 360

    return center_col, center_row, heading