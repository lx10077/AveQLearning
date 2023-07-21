import numpy as np
from tqdm import tqdm
from env import RandomMDP
from vi import vi
import matplotlib.pyplot as plt
import ray
from scipy.special import softmax


@ray.remote
def run(S, A, gamma, mdp, lr=0.75, lamba=1, algo='aql'):
    # compute true q values
    v0 = np.zeros(shape=(S,))
    if algo == 'aql':
        v_true = vi(mdp, v0, gamma, max_iter=10000)[-1]
    elif algo == "entropy":
        v_true = regularized_vi(mdp, v0, gamma, lamba, max_iter=10000)[-1]
    else:
        raise ValueError("No such algo {}!".format(algo))

    q_true = np.reshape(mdp.R + gamma * np.matmul(mdp.P, v_true), (S, A))

    q_0 = np.zeros(shape=(S, A))
    q_bar = moving_average()
    q_var = moving_variance()

    loss = []
    qs = []
    falls = []
    lengths = []

    T = int(1e4)
    p_s = mdp.generator(T)
    r_s = mdp.R.reshape((S, A))
    for t in range(T):
        # perform Q-Learning update
        if algo == "aql":
            aux = np.matmul(p_s[t, :, :], np.max(q_0, axis=1))
        elif algo == "entropy":
            Tq = regularized_max(q_0, lamba)
            aux = np.matmul(p_s[t, :, :], Tq)
        else:
            raise ValueError(algo)

        Taus = aux.reshape((S, A))
        eta = 1 / (t + 1) ** lr
        q = (1 - eta) * q_0 + eta * (r_s + gamma * Taus)
        q_0 = q

        if t >= T * 0.05:
            q_bar.update(q_0)        # update averaged iterates
            avg_q = q_bar.value()    # obtain averaged iterates
            q_var.update(avg_q)      # update averaged iterates

            # constructe CIs
            CI_l = q_bar.value() - 6.753 * np.sqrt(q_var.diag())
            CI_u = q_bar.value() + 6.753 * np.sqrt(q_var.diag())
            fall = (CI_l <= q_true) * (q_true <= CI_u)

            # Save stats
            loss.append(np.max(np.abs(avg_q - q_true)))
            # qs.append(avg_q)
            falls.append(fall)
            lengths.append(2 * 6.811 * np.sqrt(q_var.diag()))

    return np.array(loss), np.array(qs), np.array(falls), np.array(lengths)


def var_cal(mdp, gamma, lamba=None):
    S = mdp.S
    A = mdp.A
    v0 = np.zeros(shape=(S,))
    if lamba == None:
        v_true = vi(mdp, v0, gamma, max_iter=1000)[-1]
        q_true = np.reshape(mdp.R + gamma * np.matmul(mdp.P, v_true), (S, A))
        pi_true = np.argmax(q_true, axis=1)

        P_pi = np.zeros((S * A, S, A))
        for i in range(S):
            P_pi[:, i, pi_true[i]] = mdp.P[:, i]

    elif lamba > 0:
        v_true = regularized_vi(mdp, v0, gamma, lamba, max_iter=10000)[-1]
        q_true = np.reshape(mdp.R + gamma * np.matmul(mdp.P, v_true), (S, A))
        pi_true = softmax(q_true/lamba)

        P_pi = np.zeros((S * A, S, A))
        for i in range(S):
            for j in range(A):
                P_pi[:, i, j] = mdp.P[:, i] * pi_true[i, j]
    else:
        raise ValueError("Illegal lambda {}!".format(lamba))
    P_pi = P_pi.reshape((S * A, S * A))
    G = np.eye(S*A) - gamma * P_pi
    var_p = np.zeros((S*A, S*A))
    for sa in range(S*A):
        p_sa = mdp.P[sa,]
        var_sa = np.sum(p_sa * v_true**2) - (np.sum(p_sa * v_true)**2)
        var_p[sa, sa] = var_sa
    var = np.linalg.solve(G, var_p)
    var = np.linalg.solve(G, var.T).T
    return gamma*gamma*np.max(np.abs(np.diag(var)))


class moving_average(object):
    def __init__(self):
        self.average = None
        self.number = 0

    def update(self, x, weight=1.):
        if self.average is None:
            self.average = x * weight
        else:
            self.average = (self.average * self.number + x * weight) / (self.number + weight)
        self.number += weight

    def value(self):
        return self.average


class moving_variance(object):
    def __init__(self):
        self.weights = 0
        self.number = 0
        self.squre_sum = 0
        self.A = None
        self.b = None
        self.x = None
        self.shape = None

    def update(self, x, weight=1.):
        if self.shape is None:
            self.shape = x.shape
        x = x.reshape(-1, 1)

        self.weights += weight
        self.number += 1
        self.squre_sum += self.number ** 2 * weight
        x = x.reshape(-1, 1)
        if self.x is None:
            self.A = x @ x.T * self.number ** 2 * weight
            self.b = x * self.number ** 2 * weight
            self.x = np.copy(x)
        else:
            self.A += x @ x.T * self.number ** 2 * weight
            self.b += x * self.number ** 2 * weight
            self.x = np.copy(x)

    def value(self):
        return (self.A - self.x @ self.b.T - self.b @ self.x.T + self.squre_sum * self.x @ self.x.T) / self.weights / self.number**2

    def diag(self):
        return np.diag(self.value()).reshape(self.shape)


def regularized_max(q, lamba):
    pi = softmax(q/lamba, axis=1)
    ent = -(pi * np.log(pi)).sum(1)
    return (pi * q).sum(1) + lamba * ent


def regularized_vi(env, V, gamma, lamba, max_iter=10):
    R = env.R
    P_hat = env.P
    gamma = gamma
    S = env.S
    A = env.A
    V_list = [V]
    for k in range(max_iter):
        rq = R + gamma * np.matmul(P_hat, V)
        rq = rq.reshape(S, A)
        V = regularized_max(rq, lamba)
        V_list.append(V)
    return V_list
