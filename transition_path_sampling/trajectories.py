import time
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import muller

grid_size = 100
header = ['x1', 'x2']

potential_min = -150
potential_max = 150
pl = True
df = muller


def plot_contours(drift_func, data):
    x_grid, y_grid = np.meshgrid(np.linspace(np.min(data[:, 0])-0.2, np.max(data[:, 0]) + 0.2, grid_size),
                                 np.linspace(np.min(data[:, 1]) - 0.2, np.max(data[:, 1]) + 0.2, grid_size))
    z_grid = np.zeros((x_grid.shape[0], x_grid.shape[1]))
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            z_grid[i] = drift_func.get_potential(np.vstack([x_grid[i], y_grid[i]]).T)
    tics = np.linspace(potential_min, potential_max, 30)
    CS = plt.contour(x_grid, y_grid, z_grid, tics)
    plt.clabel(CS, inline=False, fontsize=10)


def run_trajectory(x_0, step_size, n, b, dim, drift_func):
    fig, ax = plt.subplots()
    x_arr = np.zeros((n, 2))
    x = x_0
    for i in tqdm(range(n)):
        x = get_next_iteration(x, step_size, b, dim, drift_func)
        x_arr[i][0] = x[0]
        x_arr[i][1] = x[1]
    if dim == 2:
        ax.scatter(x_arr[:, 0], x_arr[:, 1])
    return x_arr


def get_next_iteration(x, step_size, b, dim, drift_func):
    x_diff = np.random.normal(size=dim) * np.sqrt(2 * b ** -1 * step_size)
    # dx = -V(x_n) delta t + sqrt(2 b^-1 delta t) dw_t
    return x - drift_func.get_potential_gradient(x) * step_size + x_diff


def gen_trajectory(drift_func=muller, steps=2,plot=False):
    x_0 = np.array([0, 0])
    step_size = 10 ** -5
    n = 500000
    b = 1 / (10*np.sqrt(2))
    dim = 2

    data = run_trajectory(x_0, step_size, n, b, dim, drift_func)
    data = data[::steps]
    return data
