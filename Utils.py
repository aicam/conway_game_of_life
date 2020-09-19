import numpy as np

'''
    This method converts a binary board which 0 or 1 shows the state of life to
    a new matrix with same dimensions and fill each cell with the number of alive neighbour
    in the binary board
'''


def convert_boardtomatrix(board):
    matrix = np.zeros(shape=[board.shape[0], board.shape[1]])
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            a = []
            try:
                a.append(1 if board[i - 1][j] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i][j - 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i - 1][j - 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i + 1][j] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i][j + 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i + 1][j + 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i - 1][j + 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i + 1][j - 1] == 1 else 0)
            except IndexError:
                a.append(0)

            try:
                a.append(1 if board[i - 1][j] == 1 else 0)
            except IndexError:
                a.append(0)
