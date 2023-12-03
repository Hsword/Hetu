import os.path as osp
import sys
import matplotlib.pyplot as plt


def read_time(path):
    cur_time = -1
    if osp.exists(path):
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


def get_result(noover=False):
    if noover:
        prefix = 'noover'
    else:
        prefix = ''
    none_path = osp.join(dir_path, '{}none1.txt'.format(prefix))
    none_throughput = get_throughput(none_path, unit_batch_size)
    strats = ['dp', 'owt', 'optcnn', 'gpipe', 'pipedream', 'pipeopt']
    devices = [2, 4, 8, 16]
    results = {'linear':  {i: none_throughput * i for i in [1] + devices}}
    for key in strats:
        results[key] = {1: none_throughput}
    for i in devices:
        for st in strats:
            if dir_path == 'alexnet' and st == 'gpipe' and i == 16:
                continue
            else:
                global_batch_size = unit_batch_size * i
            file_path = osp.join(dir_path, '{}{}{}.txt'.format(prefix, st, i))
            cur_throughput = get_throughput(file_path, global_batch_size)
            if st in ('gpipe', 'pipedream', 'pipeopt'):
                new_file_path = osp.join(
                    dir_path, '{}half{}{}.txt'.format(prefix, st, i))
                new_throughput = get_throughput(
                    new_file_path, global_batch_size)
                cur_throughput = max(cur_throughput, new_throughput)
            elif st == 'optcnn':
                new_file_path = osp.join(
                    dir_path, '{}{}{}_nopix.txt'.format(prefix, st, i))
                new_throughput = get_throughput(
                    new_file_path, global_batch_size)
                cur_throughput = max(cur_throughput, new_throughput)
                new_file_path = osp.join(
                    dir_path, '{}{}{}_nonccl.txt'.format(prefix, st, i))
                new_throughput = get_throughput(
                    new_file_path, global_batch_size)
                cur_throughput = max(cur_throughput, new_throughput)
                new_file_path = osp.join(
                    dir_path, '{}{}{}_nopix_nonccl.txt'.format(prefix, st, i))
                new_throughput = get_throughput(
                    new_file_path, global_batch_size)
                cur_throughput = max(cur_throughput, new_throughput)
            results[st][i] = cur_throughput
    return results


if __name__ == '__main__':
    assert len(sys.argv) == 2
    dir_path = sys.argv[1]
    if dir_path == 'alexnet':
        unit_batch_size = 128
    else:
        unit_batch_size = 32

    # over_results = get_result()
    # # draw
    # xlabels = [1, 2, 4, 8, 16]
    # for key, value in over_results.items():
    #     cur_xlabels = []
    #     for x in xlabels:
    #         if value.get(x, -1) > 0:
    #             cur_xlabels.append(x)
    #     plt.plot(cur_xlabels, [value[x] for x in cur_xlabels], label=key)
    # plt.xlabel('# workers')
    # plt.ylabel('Troughput / # samples per second')
    # plt.title('{} overlap'.format(dir_path))
    # plt.legend()
    # plt.savefig('{}_overlap.png'.format(dir_path))
    # plt.close()

    noover_results = get_result(noover=True)
    # draw
    xlabels = [1, 2, 4, 8, 16]
    for key, value in noover_results.items():
        cur_xlabels = []
        for x in xlabels:
            if value.get(x, -1) > 0:
                cur_xlabels.append(x)
        plt.plot(cur_xlabels, [value[x] for x in cur_xlabels], label=key)
    plt.xlabel('# workers')
    plt.ylabel('Troughput / # samples per second')
    plt.title('{} no overlap'.format(dir_path))
    plt.legend()
    plt.savefig('{}_nooverlap.png'.format(dir_path))
    plt.close()

    # # draw
    # xlabels = [1, 2, 4, 8, 16]
    # for key, value in noover_results.items():
    #     cur_xlabels = []
    #     for x in xlabels:
    #         if value.get(x, -1) > 0:
    #             cur_xlabels.append(x)
    #     plt.plot(cur_xlabels, [value[x]
    #                            for x in cur_xlabels], label=key+'_noover')
    # for key, value in over_results.items():
    #     cur_xlabels = []
    #     for x in xlabels:
    #         if value.get(x, -1) > 0:
    #             cur_xlabels.append(x)
    #     plt.plot(cur_xlabels, [value[x]
    #                            for x in cur_xlabels], label=key+'_over')
    # plt.xlabel('# workers')
    # plt.ylabel('Troughput / # samples per second')
    # plt.title('{} all'.format(dir_path))
    # plt.legend()
    # plt.savefig('{}_all.png'.format(dir_path))
    # plt.close()
