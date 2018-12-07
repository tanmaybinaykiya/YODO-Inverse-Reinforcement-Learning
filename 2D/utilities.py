import pickle

import numpy as np


def euclidean_dist(point1, point2):
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))


def manhattan_distance(block_position_i, block_position_j):
    return np.abs(block_position_i[0] - block_position_j[0]) + np.abs(block_position_i[1] - block_position_j[1])


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
