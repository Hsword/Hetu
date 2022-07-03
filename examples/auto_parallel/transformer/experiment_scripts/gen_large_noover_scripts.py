
models = {
    'bert': 4,
    # 'bert': 8,
    'gpt2': 2,
}

devices = [2, 4, 8]
strategies = ['dp', 'optcnn', 'megatronlm', 'gpipe', 'pipedream']


def import_os(fw):
    print("import os", file=fw)
    print(file=fw)


def make_system(fw, string='sleep 10'):
    print("os.system('{}')".format(string), file=fw)


def make_str(fw, model, bs, dv, st, half=False, pix=True, nccl=True):
    if dv == 16:
        cur_str = 'heturun -c w16.yml '
    elif dv > 1:
        cur_str = 'heturun -w {} '.format(dv)
    else:
        assert st == 'none'
        cur_str = ''
    cur_str += 'python ../{}_main.py'.format(model)
    cur_str += ' --batch-size {}'.format(bs)
    cur_str += ' --ignore-iter 2 --log-num 10'
    cur_str += ' --strategy {}'.format(st)
    cur_str += ' --large --nooverlap'
    appen_str = ''
    if not pix:
        cur_str += ' --nopix'
        appen_str += '_nopix'
    if not nccl:
        cur_str += ' --nonccl'
        appen_str += '_nonccl'
    if half:
        cur_str += ' --batch-num-factor 2'
    half_str = 'half' if half else ''
    if st in ('optcnn', 'gpipe', 'pipedream'):
        cur_str += ' --save-path {}large/noover{}{}{}{}.json'.format(
            model, half_str, st, dv, appen_str)
    cur_str += ' > {}large/noover{}{}{}{}.txt'.format(
        model, half_str, st, dv, appen_str)
    make_system(fw, cur_str)
    make_system(fw)


def make_dir(fw, model):
    print("os.makedirs('{}large', exist_ok=True)".format(model), file=fw)
    print(file=fw)


def make_allstr(fw, st, model, bs, dv):
    if st in ('dp', 'gpipe', 'pipedream'):
        cur_bs = bs
    else:
        cur_bs = bs * dv
    make_str(fw, model, cur_bs, dv, st)
    if st == 'optcnn':
        make_str(fw, model, cur_bs, dv, st, pix=False)
        make_str(fw, model, cur_bs, dv, st, nccl=False)
        make_str(fw, model, cur_bs, dv, st, pix=False, nccl=False)
    elif st in ('gpipe', 'pipedream'):
        make_str(fw, model, cur_bs, dv, st, True)


for model, bs in models.items():
    with open('run_{}_large_noover.py'.format(model), 'w') as fw:
        import_os(fw)
        make_dir(fw, model)
        make_str(fw, model, bs, 1, 'none')
        print(file=fw)
        for dv in devices:
            for st in strategies:
                make_allstr(fw, st, model, bs, dv)
            print(file=fw)

# run all
with open('run_all_large_noover.py', 'w') as fw:
    for model in models:
        lines = open('run_{}_large_noover.py'.format(model), 'r').readlines()
        fw.writelines(lines)

dv = 16
with open('run_16_all_large_noover.py', 'w') as fw:
    import_os(fw)
    for model, bs in models.items():
        make_dir(fw, model)
        for st in strategies:
            make_allstr(fw, st, model, bs, dv)
        print(file=fw)
