import numpy as np

a = np.array([-1, -1, -6.5, 0.7])
b = np.array([0, 0, 11, 0.6])
c = np.array([-10, -10, -6.5, 0.7])
d = np.array([-200, -100, -170, 15])

x_ctr = np.array([1, 0, -0.5, -1])
y_ctr = np.array([0, 0.5, 1.5, 1])


def get_potential_gradient_vectorized(x):
    u = np.zeros_like(x)
    x_ctr_t = np.tile(x_ctr, (x.shape[0], 1))
    y_ctr_t = np.tile(y_ctr, (x.shape[0], 1))
    for i in range(4):
        v = d[i] * np.e ** (a[i] * (x[:, 0] - x_ctr_t[:, i]) ** 2 + b[i] * (x[:, 0] - x_ctr_t[:, i]) * (x[:, 1] -
                                                                                                        y_ctr_t[:, i]) +
                            c[i] * (x[:, 1] - y_ctr_t[:, i]) ** 2)
        u[:, 0] += (2 * a[i] * (x[:, 0] - x_ctr_t[:, i]) + b[i] * (x[:, 1] - y_ctr_t[:, i])) * v
        u[:, 1] += (b[i] * (x[:, 0] - x_ctr_t[:, i]) + 2 * c[i] * (x[:, 1] - y_ctr_t[:, i])) * v
    return u

def get_potential_gradient(x, sigma=0):
    x = x + np.random.randn(2) * sigma
    u = np.zeros(2)
    for i in range(4):
        v = d[i] * np.e ** (
                a[i] * (x[0] - x_ctr[i]) ** 2 + b[i] * (x[0] - x_ctr[i]) * (x[1] - y_ctr[i]) + c[i] * (x[1] - y_ctr[i]) ** 2)
        u[0] += (2 * a[i] * (x[0] - x_ctr[i]) + b[i] * (x[1] - y_ctr[i])) * v
        u[1] += (b[i] * (x[0] - x_ctr[i]) + 2 * c[i] * (x[1] - y_ctr[i])) * v
    return u


def get_potential(x):
    ret = np.zeros(x.shape[0])
    for i in range(0, 4):
        ret += d[i] * np.e ** (a[i] * (x[:, 0] - x_ctr[i]) ** 2 + b[i] * (x[:, 0] - x_ctr[i]) * (x[:, 1] - y_ctr[i]) + c[i] * (x[:, 1] - y_ctr[i]) ** 2)
    return ret


def get_score(x, beta, sigma):
    ret = 0
    for i in range(100):
        x_tilde = x + np.random.randn(2) * sigma
        ret -= get_potential_gradient(x_tilde) * beta
    return ret / 100


# should add for loop to make expectation, but it would be slow
def get_score_vectorized(x, beta, sigma):
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_tilde = x[i] + np.random.randn(2) * sigma
        ret[i] = - get_potential_gradient(x_tilde) * beta
    return ret


def get_min_A():
    return np.array([0.62347076, 0.02807048])


def get_min_B():
    return np.array([-0.55821361, 1.44174872])
