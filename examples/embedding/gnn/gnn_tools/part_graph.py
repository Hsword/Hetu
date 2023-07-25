from graphmix.partition import part_graph
from graphmix.dataset import load_dataset
from sparse_datasets import load_sparse_dataset
import argparse
import os.path as osp
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True)
    parser.add_argument("--nparts", "-n", required=True)
    parser.add_argument("--path", "-p", required=True)
    args = parser.parse_args()
    output_path = str(args.path)
    nparts = int(args.nparts)
    dataset, idx_max = load_sparse_dataset(args.dataset)
    output_path = osp.expanduser(osp.join(output_path, args.dataset))
    part_graph(dataset, nparts, output_path)
    # now write idx_max into meta.yml
    meta_file = osp.join(output_path, "meta.yml")
    with open(meta_file) as f:
        meta = yaml.load(f.read(), Loader=yaml.FullLoader)
    meta["idx_max"] = idx_max
    with open(meta_file, "w") as f:
        yaml.dump(meta, f, sort_keys=False)
