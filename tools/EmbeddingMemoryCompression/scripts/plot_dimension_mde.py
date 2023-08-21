import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


dir_path = osp.join(osp.split(osp.abspath(__file__))[0], '../datasets')


dims = {
    '64': [16,16,1,2,16,32,8,16,64,4,8,1,8,32,8,2,64,8,16,64,2,32,32,4,32,4],
    '32': [8,8,1,2,16,16,8,8,32,4,8,1,8,16,4,2,32,8,8,32,2,16,32,4,16,4],
    '16': [8,8,1,2,8,8,4,8,16,4,4,1,4,8,4,2,16,4,8,14,2,16,16,4,8,4],
}


num_embed_separate = [1460, 583, 10131227, 2202608, 305, 24, 12517, 633, 3, 93145, 5683, 8351593, 3194, 27, 14992, 5461306, 10, 5652, 2173, 4, 7046547, 18, 15, 286181, 105, 142572]


fontsize = 18

target_file = osp.join(dir_path, 'criteo', 'counter_all.bin')
cnt = np.fromfile(target_file, dtype=np.int32)
# else:
#     sparse = np.fromfile(osp.join(
#         dir_path, 'criteo', "kaggle_processed_sparse.bin"), dtype=np.int32)
#     print('Shape:', sparse.shape)
#     cnt = np.bincount(sparse).astype(np.int32)
#     indices = -np.argsort(-cnt)
#     indices.tofile(target_file)
# x = np.arange(cnt.shape[0])

colors = {'16': 'tab:blue', '32': 'tab:green', '64': 'tab:red'}
# markers = {'16': '3', '32': '4', '64': '1'}

for k, v in dims.items():
    cur_mems = np.zeros(cnt.shape, dtype=np.int32)
    offset = 0
    for nemb, vv in zip(num_embed_separate, v):
        ending = offset + nemb
        cur_mems[offset:ending] = vv
        offset = ending
    cur_mems.tofile(f'mde_{k}.bin')
    # cur_mems = np.fromfile(f'mde_{k}.bin', dtype=np.int32)
    plt.scatter(cnt, cur_mems, s=5, color=colors[k], marker='.')
    plt.savefig(f'temp_{k}.png')

plt.title('MDE Feature Dimension Distribution', fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)

plt.ylabel('# Parameters')
plt.xlabel('Popularity')
plt.savefig('dimension_mde.png')
plt.grid(linestyle='dashed', color='silver')
plt.savefig('dimension_grid_mde.png')
