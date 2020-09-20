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
            # a represents the value of 8 neighbours
            a = []
            if i - 1 >= 0:
                a.append(1 if board[i - 1][j] == 1 else 0)
            if j - 1 >= 0:
                a.append(1 if board[i][j - 1] == 1 else 0)
            if i - 1 >=0 and j - 1 >= 0:
                a.append(1 if board[i - 1][j - 1] == 1 else 0)

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
            print(a, ' ', i, ' ', j)
            matrix[i][j] = np.sum(a)
    return matrix

import numpy as np
import tensorflow as tf
from tensorflow.python import keras


def expand(x, dim, N):
    ndim = tf.shape(tf.shape(x))[0]
    expand_idx = tf.concat([tf.ones((tf.maximum(0, dim),), dtype=tf.int32), tf.reshape(N, (-1,)),
                            tf.ones((tf.minimum(ndim - dim, ndim),), dtype=tf.int32)], axis=0)
    return tf.tile(tf.expand_dims(x, dim), expand_idx)


def learned_init(units):
    return tf.Variable(initial_value=keras.initializers.glorot_uniform()(shape=(units,)))


def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return keras.initializers.truncated_normal(stddev=stddev, dtype=dtype)