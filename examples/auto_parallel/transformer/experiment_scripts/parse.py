import os.path as osp
import sys
import matplotlib.pyplot as plt


def read_time(path):
    cur_time = -1
    if not osp.exists(path):
        return cur_time
    with open(path, 'r') as fr:
        lines = fr.readlines()
    for line in lines:
        if 'each iteration time =' in line:
            cur_time = float(line.rstrip('\n').split()[-1].rstrip('ms'))
            break
    return cur_time


def get_throughput(path, batch_size):
    cur_time = read_time(path)
    return batch_size / cur_time * 1000


if __name__ == '__main__':
    assert len(sys.argv) == 2
    dir_path = sys.argv[1]
    model = dir_path[:4]
    if model == 'bert':
        unit_batch_size = 4
    else:
        unit_batch_size = 2
    none_path = osp.join(dir_path, 'noovernone1.txt')
    none_throughput = get_throughput(none_path, unit_batch_size)
    strats = ['dp', 'megatronlm', 'optcnn', 'gpipe', 'pipedream', 'pipeopt']
    devices = [2, 4, 8, 16]
    # devices = [2, 4, 8]
    results = {'linear':  {i: none_throughput * i for i in [1] + devices}}
    for key in strats:
        results[key] = {1: none_throughput}
    for i in devices:
        for st in strats:
            file_path = osp.join(dir_path, 'noover{}{}.txt'.format(st, i))
            cur_throughput = get_throughput(file_path, unit_batch_size * i)
            if st in ('gpipe', 'pipedream', 'pipeopt'):
                new_file_path = osp.join(
                    dir_path, 'nooverhalf{}{}.txt'.format(st, i))
                new_throughput = get_throughput(
                    new_file_path, unit_batch_size * i)
                cur_throughput = max(cur_throughput, new_throughput)
            elif st == 'optcnn':
                new_file_path = osp.join(
                    dir_path, 'noover{}{}_nopix.txt'.format(st, i))
                new_throughput = get_throughput(
                    new_file_path, unit_batch_size * i)
                cur_throughput = max(cur_throughput, new_throughput)
                new_file_path = osp.join(
                    dir_path, 'noover{}{}_nonccl.txt'.format(st, i))
                new_throughput = get_throughput(
                    new_file_path, unit_batch_size * i)
                cur_throughput = max(cur_throughput, new_throughput)
                new_file_path = osp.join(
                    dir_path, 'noover{}{}_nopix_nonccl.txt'.format(st, i))
                new_throughput = get_throughput(
                    new_file_path, unit_batch_size * i)
                cur_throughput = max(cur_throughput, new_throughput)
            results[st][i] = cur_throughput

    # draw
    # xlabels = [1, 2, 4, 8]
    xlabels = [1, 2, 4, 8, 16]
    for key, value in results.items():
        new_x = []
        new_y = []
        for x in xlabels:
            if value[x] > 0:
                new_x.append(x)
                new_y.append(value[x])
        plt.plot(new_x, new_y, label=key)
    plt.xlabel('# workers')
    plt.ylabel('Troughput / # samples per second')
    plt.title(model)
    plt.legend()
    plt.savefig('{}.png'.format(model))
