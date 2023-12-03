import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle


# this is the path to datasets
dir_path = osp.join(osp.split(osp.abspath(__file__))[0], '../datasets')


files = {
    '16': 'path/to/autosrh/dim16/final_ckpt',
    '32': 'path/to/autosrh/dim32/final_ckpt',
    '64': 'path/to/autosrh/dim64/final_ckpt',
}


# num_embed_separate = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]


fontsize = 18

target_file = osp.join(dir_path, 'criteo', 'counter_all.bin')
if osp.exists(target_file):
    cnt = np.fromfile(target_file, dtype=np.int32)
else:
    sparse = np.fromfile(osp.join(
        dir_path, 'criteo', "kaggle_processed_sparse.bin"), dtype=np.int32)
    print('Shape:', sparse.shape)
    cnt = np.bincount(sparse).astype(np.int32)
    cnt.tofile(target_file)

colors = {'16': 'tab:blue', '32': 'tab:green', '64': 'tab:red'}

for k, v in files.items():
    with open(v, 'rb') as fr:
        ckpt = pickle.load(fr)
    emb = ckpt['state_dict']['AutoSrhEmb_sparse']
    nparam = np.bincount(emb.row)
    if nparam.shape[0] < cnt.shape[0]:
        nparam = np.pad(nparam, (0, cnt.shape[0] - nparam.shape[0]), 'constant', constant_values=0)
    plt.scatter(cnt, nparam, s=5, color=colors[k], marker='.')
    plt.savefig(f'temp_{k}.png')

plt.title('AutoSrh Feature Density Distribution', fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)

plt.ylabel('# Parameters')
plt.xlabel('Feature ID')
plt.savefig('dimension.png')
plt.grid(linestyle='dashed', color='silver')
plt.savefig('dimension_grid.png')
