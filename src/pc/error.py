def IsWallRed(x,y,matrix):
    point = matrix[x][y]
    height = len(matrix)
    length = len(matrix[0])
    if point[0] > 0 and point[0] < length and point[1] > 0 and point[1] < length:
        midleft = matrix[x][y-1]
        midright = matrix[x][y+1]
        up = matrix[x-1][y]
        down = matrix[x+1][y]
        upleft = matrix[x-1][y-1]
        upright = matrix[x+1][y]
        downleft = matrix[x-1][y+1]
        downright = matrix[x+1][y+1]
        if downleft == 'r' or downright == 'r' or down == 'r' : return False #return x-1
        if upleft == 'r' or upright == 'r' or up == 'r' : return False # return x+1
        if midleft == 'r' or midright == 'r' : return False
    return True

def MakeWallBigger(matrix):
    RefArr = matrix.copy()
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if RefArr[x][y] == 'r':
                matrix[x][y - 1] ='r'
                matrix[x][y + 1] ='r'
                matrix[x - 1][y] ='r'
                matrix[x + 1][y] ='r'
                matrix[x + 1][y] ='r'
                matrix[x + 1][y + 1] ='r'
                matrix[x - 1][y - 1] ='r'
                matrix[x - 1][y + 1] ='r'

    return matrix