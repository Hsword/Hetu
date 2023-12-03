def readlines(path):
    return open(path, 'r').readlines()


if __name__ == '__main__':

    simu_path = 'optcnn.txt'
    num_workers = 2
    real_path = 'temp{}.txt'
    out_path = 'diff{}.txt'
    threshold = 0.5

    simu = {}
    for line in readlines(simu_path):
        ents = line.split()
        act = ents[0]
        name = ents[-1]
        t = float(ents[1])
        if act == 'comm':
            simu[act + name] = simu.get(act + name, 0) + t
        elif act == 'execute':
            simu[name] = t
        elif act == 'update':
            simu[act] = simu.get(act, 0) + t
        elif act == 'allreduce':
            simu[act + name] = simu.get(act + name, 0) + t
        else:
            assert False

    for i in range(num_workers):
        rtime = {}
        rin = {}
        for line in readlines(real_path.format(i)):
            ents = line.split()
            name = ents[0]
            t = float(ents[-1])
            inp = ents[1].lstrip('[').rstrip(']')
            rtime[name] = t
            rin[name] = inp

        ig = []
        diffs = {}
        for key in rtime:
            if key.startswith('Optimizer'):
                diff = rtime[key] - simu['update']
            elif key.startswith('AllReduceCommunicateOp'):
                diff = rtime[key] - simu['allreduce' + rin[key]]
            else:
                if key not in simu:
                    ig.append(key)
                diff = rtime[key] - simu.get(key, 0)
            if abs(diff) > threshold:
                diffs[key] = diff

        with open(out_path.format(i), 'w') as fw:
            for k, v in diffs.items():
                print(k, v, file=fw, flush=True)

        print(ig)
