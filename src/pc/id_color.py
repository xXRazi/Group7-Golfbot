def ball_pos_approx(matrix, color):

    row = len(matrix)
    col = len(matrix[0])

    color_list = []
    color_list_final = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == color:
                color_list.append((i, j))

    last_ball_pos = [(1, 0), (1, 1), (0, 1)]

    for i in color_list:
        right_check = False
        diagonal_check = False

        for dir in last_ball_pos:
            new_row = i[0] + dir[0]
            new_col = i[1] + dir[1]

            if 0 <= new_row < row and 0 <= new_col < col:
                if matrix[new_row][new_col] != color:
                    if not right_check:
                        right_check = True
                    elif not diagonal_check:
                        diagonal_check = True
                        last_id = i

        if right_check and diagonal_check:
            final_color_spot = (last_id[1]//2, last_id[0]//2)
            color_list_final.append(final_color_spot)

    return color_list_final
