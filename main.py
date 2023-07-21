import numpy as np
from tqdm import tqdm
from algo import run, var_cal
from env import RandomMDP, SimpleMDP
import matplotlib.pyplot as plt
import json
import argparse
import ray

parser = argparse.ArgumentParser(description='Polyak Q learning')
args = parser.parse_args()

alg = 'entropy'
lr = 0.51
# gammas = [0.6+0.01*i for i in range(11)]
gammas = [0.6, 0.7, 0.8, 0.9]
S = 4
A = 3
r = 5
lamba = 1
mdp = RandomMDP(S, A, r)
data = []
replica = 500
ray.init(num_cpus=50, num_gpus=0)

for gamma in tqdm(gammas):
    res = ray.get([run.remote(S, A, gamma, mdp, lr, lamba, alg) for _ in range(replica)])
    # res is a list, len = # of relica
    # For each element in res, it's a tuple, len = 3, including loss, qs, falls
    # loss shape=(Timestep, ), qs shape = (Timestep, S, A)
    # avg_loss = np.mean(losses, axis=0)
    qs = []
    loss = []
    falls = []
    lengths = []

    for i in range(replica):
        output = res[i]
        loss.append(output[0].tolist())    # loss shape = (replica, timestep)
        falls.append(output[2].tolist())   # falls shape = (replica, timestep, S, A)
        lengths.append(output[3].tolist())

    var = var_cal(mdp, gamma, lamba)
    data.append((gamma, np.log(var), loss, qs, falls, lengths))
    # data is a tuple with len=# of gammas

dct = {'data': data}
f = open('./results/S{}_A{}_Alg{}_lr{}_random_r{}_burn.json'.format(S, A, alg, lr, r), 'w')
f.write(json.dumps(dct))
f.close()

