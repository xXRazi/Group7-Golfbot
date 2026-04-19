"""
node_list = {"A": {"B":11,"C":15},"B":{"A":10,"Goal_B:5"}}
"""
import math
from collections import defaultdict
import heapq
#from camera import color_matrix
from create_test_image import test_matrix

ROW = 480
COL = 640

#ROW = 62
#COL = 110

"""

class Cell:
    def __init__(self):
        self.parent_row = color_matrix[0][0]
        self.parent_col = color_matrix[0][0]
        self.f = float('inf') #total cost (g+h)
        self.g = float('inf') #Cost from starting cell
        self.h = 0 #Heuristic cost

"""


class test_Cell:
    def __init__(self):
        self.parent_row = 0
        self.parent_col = 0
        self.f = float('inf') #total cost (g+h)
        self.g = float('inf') #Cost from starting cell
        self.h = 0 #Heuristic cost



def destination(row, col, dest_y,dest_x):
    return col == dest_x and row == dest_y

def get_h(ry, rx, by, bx):
    distance:int = ((rx-bx)**2 + (ry-by)**2)**0.5
    return distance

def trace_path(cell,dx,dy):
    path = []
    row = dy
    col = dx

    while not (cell[row][col].parent_row == row and cell[row][col].parent_col == col):
        path.append((row,col))
        temp_row = cell[row][col].parent_row
        temp_col = cell[row][col].parent_col
        row = temp_row
        col = temp_col

    path.append((row,col))
    path.reverse()

    for i in path:
        print("->", i, end=" ")
    print()

def is_valid(row,col):
    if (row >= 0) and (row < ROW) and (col >= 0) and (col < COL):
         return True
    return False


"""
Get heuristic skal have en liste af distances fra bold til robot,
så skal vi finde max i listen når vi er færdige
"""

def is_unblocked(grid, row, col):
    if grid[row][col] == '.' or grid[row][col] == 'W':
        return True

def A_star(color_matrix, src, dest):

    if not is_valid(src[0],src[1]) or not is_valid(dest[0],dest[1]):
        return "The specified rows and columns are not valid"

    if not is_unblocked(color_matrix, src[0], src[1]) or not is_unblocked(color_matrix, dest[0], dest[1]):
        return "Source or destination is blocked"

    if destination(src[0],src[1],dest[0],dest[1]):
        return "We are already there, bozo"

    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    cell_details = [[test_Cell() for _ in range(COL)] for _ in range(ROW)]


    row = src[0]
    col = src[1]
    cell_details[row][col].f = 0
    cell_details[row][col].g = 0
    cell_details[row][col].h = 0
    cell_details[row][col].parent_row = row
    cell_details[row][col].parent_col = col

    open_list = []
    heapq.heappush(open_list,(0.0,row,col))

    found_dest = False

    while len(open_list) > 0:
        p = heapq.heappop(open_list)

        row = p[1]
        col = p[2]

        closed_list[row][col] = True

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dir in directions:
            new_row = row + dir[0]
            new_col = col + dir[1]

            if is_valid(new_row,new_col) and is_unblocked(color_matrix, new_row, new_col) and not closed_list[new_row][new_col]:
                if destination(new_row, new_col, dest[0], dest[1]):
                    cell_details[new_row][new_col].parent_row = row
                    cell_details[new_row][new_col].parent_col = col
                    print("destination cell found")
                    trace_path(cell_details,dest[1],dest[0])
                    found_dest = True
                    return "You reached the destination"
                else:
                    g_new = cell_details[row][col].g + 1.4
                    h_new = get_h(new_row,new_col,dest[0],dest[1])
                    f_new = g_new + h_new

                    if cell_details[new_row][new_col].f == float('inf') or cell_details[new_row][new_col].f > f_new:
                        heapq.heappush(open_list, (f_new, new_row, new_col))
                        cell_details[new_row][new_col].f = f_new
                        cell_details[new_row][new_col].g = g_new
                        cell_details[new_row][new_col].h = h_new
                        cell_details[new_row][new_col].parent_row = row
                        cell_details[new_row][new_col].parent_col = col
    if not found_dest:
        return "You fucked up, bozo"



def test():
    # Define the grid (1 for unblocked, 0 for blocked)
    grid = [
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]
    ]

    # Define the source and destination
    src = [8, 0]
    dest = [0, 0]

    # Run the A* search algorithm
    A_star(grid, src, dest)

def vision_test(matrix, src, dest):
   return A_star(matrix, src, dest)


#print(vision_test(test_matrix, [10,10], [17,10]))

