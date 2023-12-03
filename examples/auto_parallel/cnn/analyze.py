
def readlines(path):
    return open(path, 'r').readlines()


sim = readlines('pipe.txt')
num_devs = 4

sim_time = {}
for line in sim:
    rtype, t, name = line.rstrip('\n').split()
    t = float(t)
    if rtype == 'update':
        key = 'optimizer'
        sim_time[key] = sim_time.get(key, 0) + t
    elif rtype == 'allreduce':
        key = 'allreduce'
        # assert key not in sim_time
        sim_time[key] = sim_time.get(key, 0) + t
        # sim_time[key] = t
    elif rtype == 'comm':
        key = 'comm'
        # key = 'comm_' + name
        # assert key not in sim_time
        sim_time[key] = sim_time.get(key, 0) + t
        # sim_time[key] = t
    elif rtype == 'execute':
        key = name
        assert key not in sim_time, key
        sim_time[key] = t
    else:
        assert False, line

for i in range(num_devs):
    real_times = {}
    comm = 0.
    for line in readlines('realtime{}.txt'.format(i)):
        entries = line.rstrip('\n').split()
        name = entries[0]
        t = float(entries[-1])
        if name in sim_time:
            real_times[name] = t
        elif name.startswith('Optimizer'):
            real_times['optimizer'] = t
        elif name.startswith('AllReduce'):
            real_times['allreduce'] = real_times.get('allreduce', 0) + t
        elif not name.startswith('All_time:'):
            assert name.startswith('Pipeline') or name.startswith('Sum') or name.startswith('Split') or name.startswith('Concate') or name.startswith(
                'AllReduce') or name.startswith('AllGather') or name.startswith('ReduceScatter') or name.startswith('ReduceCom') or name.startswith('BroadcastComm')
            real_times['comm'] = real_times.get('comm', 0) + t

    diff = {}
    for key in real_times:
        # if key not in real_times:
        #     assert sim_time[key] == 0, '{}, {}'.format(key, sim_time[key])
        #     continue
        real_time = real_times[key] * 2
        sim_t = sim_time.get(key, 0)
        if abs(real_time - sim_t) > 0.5:
            diff[key] = (real_time, sim_t)
    with open('diff{}.txt'.format(i), 'w') as fw:
        for key, value in diff.items():
            print(key, value, file=fw)
