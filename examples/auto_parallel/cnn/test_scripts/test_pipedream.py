import os
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-d', '--device', type=int, default=2)
    parser.add_argument('-c', '--count', type=int, default=10)
    parser.add_argument('-b', '--batchsize', type=int, default=32)
    args = parser.parse_args()

    def get_simulate_time(path, index=1):
        with open(path, 'r') as fr:
            lines = fr.readlines()
        sim_time = float(lines[index].split()[-1].rstrip('ms.\n'))
        for line in lines:
            if 'each iteration time =' in line:
                real_time = float(line.split()[-1].rstrip('ms\n'))
                break
        return sim_time, real_time

    os.chdir('../')
    os.system('heturun -w {} python gen_configs.py --model {} --count {} --pipedream > test_strategy/useless.txt'.format(
        args.device, args.model, args.count))
    os.system('heturun -w {} python main.py --model {} --strategy pipedream --noover --batch-size {} --log-num 2 --save-path test_strategy/result.json > test_strategy/temp.txt'.format(
        args.device, args.model, args.batchsize))
    os.system('heturun -w {} python main.py --model {} --strategy pipedream --noover --batch-size {} --log-num 2 --load-path test_strategy/result.json > test_strategy/temptest.txt'.format(
        args.device, args.model, args.batchsize))
    std_sim_time, _ = get_simulate_time(
        'test_strategy/temp.txt', index=args.device+1)
    sim_time, _ = get_simulate_time(
        'test_strategy/temptest.txt', index=args.device+1)
    assert abs(
        sim_time - std_sim_time) < 1e-3, 'Got time {} and {}.'.format(sim_time, std_sim_time)
    for i in range(args.count):
        try:
            os.system('heturun -w {} python main.py --model {} --strategy pipedream --noover --batch-size {} --log-num 20 --load-path test_strategy/config{}.json  > test_strategy/temp{}.txt'.format(
                args.device, args.model, args.batchsize, i, i))
            cur_sim_time, _ = get_simulate_time(
                'test_strategy/temp{}.txt'.format(i))
        except:
            # considering out of memory
            continue
        assert cur_sim_time >= sim_time
