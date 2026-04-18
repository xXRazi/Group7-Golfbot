def id_color(matrix, color):

    row = len(matrix)
    col = len(matrix[0])

    color_list = []

    for i in range(row):
        for j in range(col):
            if matrix[i][j] == color:
                color_list.append((i, j))

    return color_list
