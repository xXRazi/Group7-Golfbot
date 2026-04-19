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
                    if abs(k[0] - j[0]) <= 1 and abs(k[1] - j[1]) <= 1:
                        same_ball.append(j)
                        used.append(j)
                        changed = True
                        break

        last_pixel = max(same_ball, key=lambda p: (p[0], p[1]))
        color_list_final.append(last_pixel)

    return color_list_final