from corner_lineup import (
    create_corner_lineups,
    create_corner_boxes,
    ball_in_corner
)

ROWS = 40
COLS = 60

test_matrix = [["." for _ in range(COLS)] for _ in range(ROWS)]

for c in range(COLS):
    test_matrix[0][c] = "R"
    test_matrix[ROWS-1][c] = "R"

for r in range(ROWS):
    test_matrix[r][0] = "R"
    test_matrix[r][COLS-1] = "R"

# Left goal tape
for r in range(15, 25):
    test_matrix[r][5] = "b"

# Right goal tape
for r in range(15, 25):
    test_matrix[r][54] = "P"

center_r = 20
center_c = 30

for r in range(center_r - 3, center_r + 4):
    test_matrix[r][center_c] = "G"

for c in range(center_c - 3, center_c + 4):
    test_matrix[center_r][c] = "G"

test_matrix[2][2] = "W"      # corner ball
test_matrix[37][57] = "W"    # corner ball
test_matrix[18][31] = "W"    # obstacle ball

def print_matrix(matrix):

    for row in matrix:
        print("".join(row))

print_matrix(test_matrix)

corner_lineups = create_corner_lineups(test_matrix)

print(corner_lineups)

corner_boxes = create_corner_boxes(test_matrix)

print(corner_boxes)

print("\nTesting balls:")

for ball in [(2,2), (37,57), (18,31)]:

    result = ball_in_corner(ball, corner_boxes)

    if result:

        print(
            f"Ball {ball} is in {result}, "
            f"lineup point = {corner_lineups[result]}"
        )

    else:
        print(f"Ball {ball} is not in a corner")