import csv

import numpy as np
from matplotlib import pyplot as plt


def interpolate(index, arr):
    if int(np.floor(index) + 1) >= arr.shape[0]:
        return arr[int(np.floor(index))]
    return (index - np.floor(index)) * arr[int(np.floor(index))] + (1 - index + np.floor(index)) * arr[int(np.floor(index) + 1)]


def standardize(data=None):
    tp_length = 500
    tps = [np.array(sublist) for sublist in data]

    adj_tps = np.zeros((len(tps), tp_length+1, 2))
    np.set_printoptions(suppress=True)
    for i in range(len(tps)):
        t = len(tps[i]) * 10 ** -5
        adj_tps[i][tp_length] = np.array([t, t])
        for j in range(0, tp_length):
            j_new = j * tps[i].shape[0] / tp_length
            new_point = interpolate(j_new, tps[i])
            adj_tps[i][j] = new_point
    adj_tps[:,:-1,0] = (adj_tps[:,:-1,0] + np.pi - 1.5) % (2 * np.pi) - np.pi
    adj_tps = np.round(adj_tps, 4)
    return adj_tps