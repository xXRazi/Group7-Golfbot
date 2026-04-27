def ball_pos_approx(matrix, color):
    row = len(matrix)
    col = len(matrix[0])

    color_list = []
    candidate_list = []
    color_list_final = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == color:
                color_list.append((i, j))

    last_ball_pos = [(1, 0), (1, 1), (0, 1)]

    # Find candidate last-pixels
    for i in color_list:
        checks = 0

        for dir in last_ball_pos:
            new_row = i[0] + dir[0]
            new_col = i[1] + dir[1]

            if 0 <= new_row < row and 0 <= new_col < col:
                if matrix[new_row][new_col] != color:
                    checks += 1
            else:
                checks += 1

        if checks == 3:
            candidate_list.append(i)

    used = []

    # Group nearby candidates transitively
    for i in candidate_list:
        if i in used:
            continue

        same_ball = [i]
        used.append(i)

        changed = True
        while changed:
            changed = False

            for j in candidate_list:
                if j in used:
                    continue

                for k in same_ball:
                    if abs(k[0] - j[0]) <= 4 and abs(k[1] - j[1]) <= 4:
                        same_ball.append(j)
                        used.append(j)
                        changed = True
                        break

        last_pixel = max(same_ball, key=lambda p: (p[0], p[1]))
        color_list_final.append(last_pixel)

    return color_list_final

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

