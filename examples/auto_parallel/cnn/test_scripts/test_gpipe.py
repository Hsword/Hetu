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
    os.chdir('../')
    os.makedirs('test_strategy', exist_ok=True)
    os.system('heturun -w {} python main.py --model {} --strategy gpipe --batch-size {} --log-num 5 > test_strategy/temp.txt'.format(
        args.device, args.model, args.batchsize))
    with open('test_strategy/test.txt', 'r') as fr:
        lines = fr.readlines()
    accum = [float(x) for x in lines[0].rstrip('\n').split()]
    num_workers, num_group = [int(x) for x in lines[1].rstrip('\n').split()]
    assert num_workers == args.device
    points = [int(x) for x in lines[2].rstrip('\n').split()]
    result = float(lines[3].rstrip('\n'))

    # test
    def get_result(temp_points):
        cur_start = -1
        cur_res = 0.
        for point in temp_points:
            cur_res += (accum[point+1] - accum[cur_start+1]) ** 2
            cur_start = point
        assert cur_start + 1 == num_group
        return cur_res

    assert get_result(points) == result
    print('Result: {}; Points: {}'.format(result, points))

    for i in range(args.count):
        new_points = sorted(random.sample(
            range(0, num_group-1), num_workers-1)) + [num_group - 1]
        new_result = get_result(new_points)
        assert new_result >= result
        print('Test result: {}; Test points: {}'.format(new_result, new_points))
