import numpy as np
from tqdm import tqdm
from algo import run, var_cal
from env import RandomMDP, SimpleMDP
import matplotlib.pyplot as plt
import json
import argparse
import ray

parser=argparse.ArgumentParser(description='Robust Value Iteration (sa)')
parser.add_argument('--lr', default=0.5, type=float, help='poly index')
parser.add_argument('--avg', action='store_true')
parser.add_argument('--rescale', action='store_true')
args = parser.parse_args()

#gammas = [0.6+0.01*i for i in range(31)]
gammas = [0.9]
S = 5
A = 2
data = []
lr = args.lr
ray.init(num_cpus=56, num_gpus=0)

for gamma in tqdm(gammas):
    mdp = SimpleMDP((4*gamma-1)/(3*gamma))
    losses = np.array(ray.get([run.remote(S, A, gamma, mdp, lr=lr, avg=args.avg, rescale=args.rescale) for _ in range(int(1e3))]))
    avg_loss = np.mean(losses, axis=0)
    var = var_cal(mdp, gamma)
    data.append((np.log(1/(1-gamma)), np.log(var), avg_loss.tolist()))

dct = {'data': data}
f = open('./S:{}_A:{}_lr:{}_Avg:{}_Rescale:{}_simple.json'.format(S, A, lr, args.avg, args.rescale), 'w')
f.write(json.dumps(dct))
f.close()

