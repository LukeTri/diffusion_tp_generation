import csv
import numpy as np
from tqdm import tqdm

import muller


def rec_transition_paths(data=None):
    radius = 0.1
    tps = []
    min_A = np.array([0.62347076, 0.02807048])
    min_B = np.array([-0.55821361, 1.44174872])
    temp_path = []

    arr = np.array(data).astype(float)
    for i in tqdm(range(arr.shape[0])):
        if np.linalg.norm(arr[i] - min_A) <= radius:
            from_A = True
            temp_path = []
        elif np.linalg.norm(arr[i] - min_B) <= radius:
            if from_A:
                arr[i] = arr[i].round(4)
                temp_path.append(arr[i])
                print(len(temp_path))
                tps.append(temp_path)
                temp_path = []
            from_A = False
        arr[i] = arr[i].round(4)
        temp_path.append(arr[i])
    return tps
