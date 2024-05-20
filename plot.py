import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
from src.utils.constants import EVENTS

EXPS_NUM = np.arange(1,43)
RANDOM_BASE = 0
    #EXPS_LABELS = ["random", "unc", "unc_kmeans", "bald_kmeans", "bald_base"]
EXPS_LABELS = EXPS_NUM.copy().astype(str)

SAMPLES_NUM = 10
RUNS_NUM = 5
RES = "/hahomes/jnascimento/exps/2024-bnn-al/results/{}/al_isel/{}/{}_{}_{}.json"
X = [18, 34, 50]
# X = [450]
# X = np.arange(6,51,4)
K = 3

fig, ax_base = plt.subplots(2,4, figsize=(14,6), dpi=300)
fig.delaxes(ax_base[1][3])

res_vec = np.zeros([len(EXPS_NUM), len(EVENTS), len(X), SAMPLES_NUM, RUNS_NUM])

for ixe, e in enumerate(EVENTS):
    plt_x = ixe // 4
    plt_y = ixe % 4
    ax = ax_base[plt_x, plt_y]

    for k, exp in enumerate(EXPS_NUM):
        for i, s in enumerate(X):
            for j in range(SAMPLES_NUM):
                for r in range(RUNS_NUM):
                    stats_file = RES.format(e, exp, s, j, r)
        
                    if os.path.isfile(stats_file):
                        with open(stats_file, "r") as fp:
                            data = json.load(fp)
            
                        res_vec[k][ixe][i][j][r] = data["f1_test"]
    
res_vec[res_vec == 0] = np.nan
f1 = np.nanmean(res_vec, axis=(3,4))
std = np.nanstd(res_vec, axis=(3,4))

f1_gen = np.nanmean(res_vec, axis=(1,3,4))
std_gen = np.nanstd(res_vec, axis=(1,3,4))

f1_sum = np.sum(f1_gen, axis=1)
for i, s in enumerate(f1_sum):
    if np.isnan(s):
        f1_sum[i] = 0

highest_indices = np.argsort(f1_sum)[-1*K:]

highest_indices = np.concatenate([highest_indices, [RANDOM_BASE]])

for ixe, e in enumerate(EVENTS):
    plt_x = ixe // 4
    plt_y = ixe % 4
    ax = ax_base[plt_x, plt_y]

    for k in highest_indices:
        ax.plot(X, f1[k][ixe], linestyle='-', marker='o', label="{}".format(EXPS_LABELS[k]))

        ax.set_xticks(X)
        ax.set_yticks(np.arange(0.40,0.90,0.1))
        ax.set_title(e)

        ax.legend(loc='lower right')

        ax.grid()
    
handles, labels = ax_base[0][0].get_legend_handles_labels()

for i in range(2):
    for j in range(4):
        ax_base[i][j].legend_ = None

fig.legend(handles, labels, loc='lower right')

fig.savefig("plots/al_comparisons.png")
plt.close(fig)

for i, s in enumerate(f1_sum):
    print(i+1, f1_sum[i])

for k in highest_indices:
    plt.plot(X, f1_gen[k], linestyle='-', marker='o', label="{}".format(EXPS_LABELS[k]))
    print(EXPS_NUM[k], EXPS_LABELS[k], np.round(f1_sum[k],2), np.round(f1_gen[k], 2), np.round(std_gen[k], 2))

plt.title("AL - Mean")
plt.xticks(X)
plt.yticks(np.arange(0.40, 0.90, 0.1))
plt.legend()

plt.savefig("plots/al_comparisons_mean.png")


    



