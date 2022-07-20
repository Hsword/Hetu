# make scripts for run_compressed.py

methods = [
    # 'full',
    'hash',
    'compo',
    'learn',
    'dpq',
    # 'autodim',
]

ectxs = [
    -1,  # cpu
    0,  # gpu
]

bss = [
    128,
    # 256,
    # 512,
    # 1024,
    # 2048,
    # 4096,
    # 8192,
    # 16384,
]

opts = [
    'sgd',
    'adagrad',
    'adam',
    'amsgrad',
]

dim = 16
# dim = 8
# dim = 32
# dim = 128

lrs = [
    0.1,
    0.01,
    # 0.001,
    # 0.0001,
]

nep = 700

lines = []
ctxs = [4, 6, 7]
ind = 0

for met in methods:
    for op in opts:
        for bs in bss:
            for lr in lrs:
                sep = '&' if ind != len(ctxs) - 1 else '\n'
                ctx = ctxs[ind]
                lines.append(
                    f'python run_compressed.py --method {met} --bs {bs} --opt {op} --dim {dim} --lr {lr} --val --all --nepoch {nep} --ctx {ctx} {sep}\n'
                )
                ind = (ind + 1) % len(ctxs)

with open('scripts.sh', 'w') as fw:
    fw.writelines(lines)
