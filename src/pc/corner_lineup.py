# corner_lineup.py


def find_center(matrix, color):
    positions = []

    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == color:
                positions.append((r, c))

    if not positions:
        return None

    return (
        sum(r for r, c in positions) // len(positions),
        sum(c for r, c in positions) // len(positions)
    )


def find_corners(matrix):

    reds = []

    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if matrix[r][c] == "R":
                reds.append((r, c))

    if not reds:
        return None

    top_left = min(reds, key=lambda p: p[0] + p[1])
    top_right = min(reds, key=lambda p: p[0] - p[1])
    bottom_left = max(reds, key=lambda p: p[0] - p[1])
    bottom_right = max(reds, key=lambda p: p[0] + p[1])

    return {
        "TL": top_left,
        "TR": top_right,
        "BL": bottom_left,
        "BR": bottom_right
    }


def distance_to_left_wall(matrix, tape_pos):

    row, col = tape_pos

    dist = 0

    while col - dist >= 0:

        if matrix[row][col - dist] == "R":
            return dist

        dist += 1

    return None


def distance_to_right_wall(matrix, tape_pos):

    row, col = tape_pos

    dist = 0

    while col + dist < len(matrix[0]):

        if matrix[row][col + dist] == "R":
            return dist

        dist += 1

    return None


def create_corner_lineups(matrix, tape_color="b"):

    corners = find_corners(matrix)

    if corners is None:
        return None

    left_tape = find_center(matrix, "b")
    right_tape = find_center(matrix, "P")

    left_offset = distance_to_left_wall(
        matrix,
        left_tape
    )

    right_offset = distance_to_right_wall(
        matrix,
        right_tape
    )

    corners = find_corners(matrix)

    lineups = {

        "TL": (
            corners["TL"][0] + left_offset,
            corners["TL"][1] + left_offset
        ),

        "BL": (
            corners["BL"][0] - left_offset,
            corners["BL"][1] + left_offset
        ),

        "TR": (
            corners["TR"][0] + right_offset,
            corners["TR"][1] - right_offset
        ),

        "BR": (
            corners["BR"][0] - right_offset,
            corners["BR"][1] - right_offset
        )
    }

    print("Left tape:", left_tape)
    print("Right tape:", right_tape)

    print("Left offset:", left_offset)
    print("Right offset:", right_offset)

    return lineups


def create_corner_boxes(matrix, tape_color="b"):
    corners = find_corners(matrix)

    left_tape = find_center(matrix, "b")
    right_tape = find_center(matrix, "P")

    left_offset = distance_to_left_wall(matrix, left_tape)
    right_offset = distance_to_right_wall(matrix, right_tape)

    box_size = (left_offset + right_offset) // 2

    return {

        "TL": (
            corners["TL"][0],
            corners["TL"][1],
            corners["TL"][0] + box_size,
            corners["TL"][1] + box_size
        ),

        "TR": (
            corners["TR"][0],
            corners["TR"][1] - box_size,
            corners["TR"][0] + box_size,
            corners["TR"][1]
        ),

        "BL": (
            corners["BL"][0] - box_size,
            corners["BL"][1],
            corners["BL"][0],
            corners["BL"][1] + box_size
        ),

        "BR": (
            corners["BR"][0] - box_size,
            corners["BR"][1] - box_size,
            corners["BR"][0],
            corners["BR"][1]
        )
    }


def ball_in_corner(ball_pos, corner_boxes):

    row, col = ball_pos

    for corner_name, box in corner_boxes.items():

        r1, c1, r2, c2 = box

        if r1 <= row <= r2 and c1 <= col <= c2:
            return corner_name

    return None