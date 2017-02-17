def sudoku(table, depth=0):
    solve = False
    row, col = -1, -1
    candidates = None
    num_steps = depth

    for i in range(9):
        for j in range(9):
            if table[i][j] == 0:
                new_candidate = check_candidates(table, i, j)
                if row < 0 or len(new_candidate) < len(candidates):
                    row, col = i, j
                    candidates = new_candidate

    if row < 0:
        solve = True
    else:
        for candidate in candidates:
            table[row][col] = candidate
            success, n_steps = sudoku(table, depth + 1)
            if success:
                solve = True
                num_steps = n_steps
                break

            table[row][col] = 0

    return solve, num_steps

def check_candidates(table, row, col):
    collision = []
    # number = 1

    for number in range(1, 10):
        is_collision = False

        for i in range(9):
            if (table[row][i] == number or
                table[i][col] == number or
                table[(row - row % 3) + i/3][(col - col % 3) + i % 3] == number):
                is_collision = True
                break

        if is_collision != True:
            collision.append(number)

    return collision

def main():
    test_number = [[0,0,0,7,0,0,0,9,0],
                   [0,0,9,0,3,0,0,8,0],
                   [8,0,0,2,0,0,4,7,6],
                   [1,0,0,0,0,5,8,0,0],
                   [0,2,0,0,1,0,0,6,0],
                   [0,0,8,0,0,0,0,0,9],
                   [6,1,4,0,0,7,0,0,8],
                   [0,8,0,0,4,0,6,0,0],
                   [0,7,0,0,0,2,0,0,0]]

    import numpy as np

    res, steps = sudoku(test_number)
    print "Solved: {0} in {1} steps".format(res, steps)
    print np.matrix(test_number)

if __name__ == '__main__':
    main()
