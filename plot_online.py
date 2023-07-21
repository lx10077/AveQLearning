import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax1 = ax.twinx()

ax.set_xlabel(r'$T$')
ax.set_ylabel(r'Coverage rate (%)')
ax.set_ylim([60, 100])
ax.axhline(y=95, color='k', linestyle='--')

ax1.set_ylabel(r'CI length ($10^{-2}$)')
ax1.set_ylim([0, 10])


algo = "aql"

s = 0
a = 0

fs = os.listdir('./results/')

def whether_file(name):
    if "DS_Store" in name:
        return False
    else:
        return True


fs = filter(whether_file, fs)
asasa = "S/4_A/3_Alg/aql_lr/0.51_random.json"
tracted_data = {}

for algo in ["aql"]:
    try:
        f = json.load(open('./results/S4_A3_Alg{}_lr0.51_random_r5_burn.json'.format(algo), 'r'))
    except:
        pass
    data = f['data']

    for i in tqdm(range(len(data))):
        gamma, _, loss, _, falls, lengths = data[i]
        if gamma == 0.9:
            continue
        falls = np.array(falls)
        lengths = np.array(lengths)
        y = falls.mean(axis=0)[:, s, a] * 100
        x = np.arange(len(y))
        ax.plot(x, y, label=r'$\gamma=${}'.format(gamma), linestyle="-")

        z = lengths.mean(axis=0)[:, s, a] * 100
        ax1.plot(x, z, label=r'$\gamma=${}'.format(gamma), linestyle='--')

        print(gamma, y[-1], z[-1])

    plt.legend()
    plt.tight_layout()
    plt.savefig('./online_r5_burn_repli500_{}_without9.pdf'.format(algo), dpi=200)
