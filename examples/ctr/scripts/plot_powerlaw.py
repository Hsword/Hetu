import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


dir_path = osp.join(osp.split(osp.abspath(__file__))[0], '../datasets')
datasets = ['avazu', 'criteo']
num_sparse = [22, 26]


fig, axs = plt.subplots(1, 2, figsize=(13.5, 6))
fontsize = 18

for i, (d, n) in enumerate(zip(datasets, num_sparse)):
    target_file = osp.join(dir_path, d, 'sort_freq.bin')
    if osp.exists(target_file):
        cnt = np.fromfile(target_file, dtype=np.int32)
    else:
        sparse = np.fromfile(osp.join(
            dir_path, d, "kaggle_processed_sparse.bin"), dtype=np.int32)
        print('Shape:', sparse.shape)
        cnt = np.bincount(sparse).astype(np.int32)
        cnt = -np.sort(-cnt)
        cnt.tofile(target_file)
    x = np.arange(cnt.shape[0])

    ax = axs[i]
    ax.scatter(x, cnt, s=5, color='tab:blue', marker='.')
    ax.set_title(d.capitalize(), fontsize=fontsize)
    ax.grid(linestyle='dashed', color='silver')
    ax.tick_params(labelsize=fontsize-2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_off()
    ax.set_xticks([1,1e1,1e2,1e3,1e4,1e5,1e6,1e7])
    ax.set_xticklabels(['$10^{0}$','$10^{1}$','$10^{2}$','$10^{3}$','$10^{4}$','$10^{5}$','$10^{6}$','$10^{7}$'])
    # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

fig.text(0.5, 0.005, 'Feature ID', ha='center', fontsize=fontsize)
fig.text(0.065, 0.5, 'Popularity', va='center', rotation='vertical', fontsize=fontsize)
# plt.ylabel('Popularity')
# plt.xlabel('Feature ID')
plt.savefig('powerlaw.png')
