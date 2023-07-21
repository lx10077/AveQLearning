import numpy as np
from tqdm import tqdm
from env import RandomMDP


def vi(env, V, gamma, max_iter=10):
    R = env.R
    P_hat = env.P
    gamma = gamma
    S = env.S
    A = env.A
    V_list = [V]
    for k in range(max_iter):
        rq = R + gamma * np.matmul(P_hat, V)
        rq = rq.reshape(S, A)
        V = rq.max(axis=1)
        V_list.append(V)
    return V_list