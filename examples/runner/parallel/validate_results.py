import numpy as np
import os.path as osp
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', default=None)
    args = parser.parse_args()

    directory = 'results'
    base = np.load(osp.join(directory, 'base.npy'))
    print('Ground truth:', base)
    for i in range(int(args.number)):
        res = np.load(osp.join(directory, 'res%d.npy' % i))
        np.testing.assert_allclose(base, res, rtol=1e-6)
        print('Result id %d passed test.' % i, res)
