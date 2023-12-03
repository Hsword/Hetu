

import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

fontsize = 18

memory_budgets = [1.00E-05, 1.00E-04, 2.50E-04, 5.00E-04, 1.00E-03, 2.50E-03, 5.00E-03, 1.00E-02, 2.50E-02, 5.00E-02, 0.1, 0.25, 0.5]
losses = [0.454, 0.4544, 0.4552, 0.4544, 0.4526, 0.4505, 0.4495, 0.4492, 0.4521, 0.4489, 0.4489, 0.45, 0.4487]
aucs = [0.8022, 0.8021, 0.8015, 0.8016, 0.8033, 0.8054, 0.8067, 0.8073, 0.8073, 0.8081, 0.8082, 0.8086, 0.8083]


plt.plot(memory_budgets, losses, color='tab:blue')
plt.title('Loss vs Memory', fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)
plt.ylabel('Test Loss')
plt.xlabel('Memory Budget')
plt.grid(linestyle='dashed', color='silver')
plt.xscale('log')
# plt.yscale('log')
plt.savefig('loss_law_normal.png')
plt.close()

plt.plot(memory_budgets, aucs, color='tab:blue')
plt.title('AUC vs Memory', fontsize=fontsize)
plt.tick_params(labelsize=fontsize-2)
plt.ylabel('Test AUC')
plt.xlabel('Memory Budget')
plt.grid(linestyle='dashed', color='silver')
plt.xscale('log')
# plt.yscale('log')
plt.savefig('auc_law_normal.png')

