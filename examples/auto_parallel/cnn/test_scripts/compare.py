import argparse

if __name__ == "__main__":
    # please set logNode in executor and log_task in flexflow
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--left', type=str, required=True)
    parser.add_argument('-r', '--right', type=str, required=True)
    args = parser.parse_args()

    def readlines(path):
        with open(path, 'r') as fr:
            return fr.readlines()

    lines1 = readlines(args.left)
    lines2 = readlines(args.right)
    temp = lines1[0].split()[0]
    if len(temp) > 2 and temp[-2] == '_':
        lines1, lines2 = lines2, lines1
    finished = False
    for l1, l2 in zip(lines1, lines2):
        assert not finished
        if l2.startswith('update'):
            assert l1.startswith('Optimizer')
            finished = True
        elif l2.startswith('split'):
            assert l1.startswith('SplitOp')
        elif l2.startswith('comm_'):
            assert l1.startswith('Pipeline')
        elif l2.startswith('concatenate_'):
            assert l1.startswith('ConcatenateOp')
        elif l2.startswith('sum_duplicate'):
            assert l1.startswith('SumOp')
        elif l2.startswith('allreduce'):
            assert l1.startswith('AllReduceCommunicateOp')
        else:
            l1node = l1.split()[0]
            l2node = l2.split()[0]
            assert l2node[:-1] == l1node + '_', '{}, {}'.format(l1node, l2node)
    assert finished
