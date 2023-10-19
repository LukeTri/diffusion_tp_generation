import torch
import numpy as np


def get_times(sig_min, const, n, beta):
    t = sig_min
    t_vals = torch.zeros(n+1)
    for i in range(0,n+1):
        t_vals[i] = t
        h = min(1, const * (1 - np.exp(-t * 2 * beta)) ** 2)
        t = t + h
    vals = torch.zeros((t_vals.shape[0]-1, 3)).float()
    for i in range(t_vals.shape[0]-1):
        vals[i][0] = (t_vals[i+1] + t_vals[i])/2
        vals[i][1] = t_vals[i+1] - t_vals[i]
        vals[i][2] = (1 - np.exp(-(t_vals[i] + t_vals[i+1]) * beta)) / (2 * beta)
    return vals


def reshape_ts(x, nlen):
    t = x[:, -1]
    ret = torch.zeros((x.shape[0], nlen + 1, 2))
    ret[:, -1] = t
    inds = torch.linspace(0, x.shape[1] - 2, nlen, dtype=int)
    for i in range(nlen):
        ind = inds[i]
        ret[:, i] = x[:, ind]
    return ret

def reshape_ts_np(x, nlen):
    t = x[:, -1]
    ret = np.zeros((x.shape[0], nlen + 1, 2))
    ret[:, -1] = t
    inds = np.linspace(0, x.shape[1] - 2, nlen, dtype=int)
    for i in range(nlen):
        ind = inds[i]
        ret[:, i] = x[:, ind]
    return ret
