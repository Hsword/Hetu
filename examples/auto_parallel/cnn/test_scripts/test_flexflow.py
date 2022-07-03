import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-c', '--count', type=int, default=10)
    args = parser.parse_args()
    os.chdir('../')
    os.system(
        'heturun -w 2 python gen_configs.py --count {} --model {}'.format(args.count, args.model))
    for i in range(args.count):
        os.system('heturun -w 2 python main.py --model {} --load-path test_strategy/config{}.json --strategy flexflow > test_strategy/temp{}.txt'.format(args.model, i, i))
        # check task graph correctness
        result0 = os.system(
            'python test_scripts/compare.py -l real_graph0.txt -r task_graph_0.txt')
        assert result0 == 0
        result1 = os.system(
            'python test_scripts/compare.py -l real_graph1.txt -r task_graph_1.txt')
        assert result1 == 0
        print('Pass test {}.'.format(i))
    real_to_sim_map = {}
    # manually check the effect of the simulator
    for i in range(args.count):
        with open('test_strategy/temp{}.txt'.format(i), 'r') as fr:
            lines = fr.readlines()
        sim_time = float(lines[1].split()[-1].rstrip('ms.\n'))
        for line in lines:
            if 'each iteration time =' in line:
                real_time = float(line.split()[-1].rstrip('ms\n'))
                break
        real_to_sim_map[real_time] = sim_time
    real_to_sim_list = sorted(list(real_to_sim_map.items()), key=lambda x: x[0])
    print(real_to_sim_list)
