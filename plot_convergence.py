import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

fig, ax = plt.subplots()
plt.xlabel(r'$\log \|\|\frac{1}{1-\gamma}\|\|_\infty$')
plt.ylabel(r'$\log T(\varepsilon, \gamma)$')

fs = os.listdir('./results/')
for fn in fs:
    f = json.load(open('./results/'+fn, 'r'))
    chars = fn.split('_')
    lr = chars[2]
    data = f['data']
    log_H = []
    errors = []
    log_t = []
    for i in range(len(data)):
        x, _, z, _ = data[i]
        log_H.append(x)
        z = np.array(z)
        z = np.mean(z, axis=0)
        errors.append(z)
        log_t.append(np.log(np.min(np.where(z<=np.exp(-4)))))
    log_H = np.array(log_H)
    errors = np.array(errors)
    log_t = np.array(log_t)

    model = LinearRegression()
    model.fit(log_H.reshape(-1,1), log_t.reshape(-1,1))
    coef = model.coef_[0][0]
    print(lr + str('_') + 'Regression Coefficient: ' + str(coef))
    log_t_pred = model.predict(log_H.reshape(-1,1))
    label = r'$k=${:.2f}'.format(coef)
    ax.plot(log_H, log_t, label=label)
ax.plot(log_H, 3*log_H+3, label=r'Baseline,$k=${}'.format(3), linestyle='dotted', color='black')
handles, labels = plt.gca().get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels)
plt.tight_layout()
plt.savefig('./AvgQ.pdf')

