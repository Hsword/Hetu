import graphmix
from graphmix.dataset import load_dataset
import numpy as np
import os.path as osp


class AmazonSparseDataset():
    def __init__(self, dataset_root):
        self.name = "AmazonSparse"
        data = np.load(osp.join(dataset_root, "graph.npz"))
        feat = np.load(osp.join(dataset_root, "sparsefeature.npy"))
        num_nodes = feat.shape[0]
        edge = data['edge'].T
        directed = np.concatenate([edge, edge[[1, 0]]], axis=1)
        self.idx_max = np.max(feat) + 1
        node_id = np.arange(num_nodes).reshape(-1, 1) + self.idx_max
        self.idx_max += num_nodes
        self.x = np.empty([num_nodes, 0])
        self.y = np.concatenate(
            [feat, node_id, data['y'].reshape(-1, 1)], axis=-1)
        self.train_mask = data["train_map"]
        self.graph = graphmix.Graph(
            edge_index=directed,
            num_nodes=num_nodes
        )
        self.num_classes = int(np.max(data['y']) + 1)


class OGBNmagDataset():
    def __init__(self, dataset_root):
        self.name = "ogbn-mag"
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=self.name, root=dataset_root)
        data = dataset[0]
        year = data.node_year['paper'].numpy()
        self.train_mask = year < 2018
        edge = data.edge_index_dict['paper', 'cites', 'paper'].numpy()
        directed = np.concatenate([edge, edge[[1, 0]]], axis=1)
        num_nodes = data.num_nodes_dict['paper']
        self.graph = graphmix.Graph(
            edge_index=directed,
            num_nodes=num_nodes
        )
        self.num_classes = dataset.num_classes

        def process_sparse_idx(rel, length, base):
            sp_idx = [[] for i in range(num_nodes)]
            for i, j in rel.T:
                sp_idx[i].append(j)
            for i in range(num_nodes):
                if len(sp_idx[i]) > length:
                    sp_idx[i] = sp_idx[i][0:length]
                while len(sp_idx[i]) < length:
                    sp_idx[i].append(-1)
            sp_idx = np.array(sp_idx)
            sp_idx += (base + 1)
            return sp_idx

        node_id = np.arange(num_nodes).reshape(-1, 1)
        field = data.edge_index_dict[(
            'paper', 'has_topic', 'field_of_study')].numpy()
        paper_field = process_sparse_idx(field, 10, num_nodes)
        idx_max = num_nodes + data.num_nodes_dict['field_of_study'] + 1
        author = data.edge_index_dict[('author', 'writes', 'paper')].numpy()
        paper_author = process_sparse_idx(author[[1, 0]], 10, idx_max)
        idx_max += data.num_nodes_dict['author'] + 1
        self.idx_max = idx_max
        self.x = np.empty([num_nodes, 0])
        self.y = np.concatenate([
            paper_field, paper_author, node_id, data.y_dict["paper"].numpy()
        ], axis=1)


def load_sparse_dataset(name):
    root_dir = osp.expanduser(osp.join('~/.graphmix_dataset/', name))
    if name == "Reddit":
        dataset = load_dataset(name)
        idx_max = dataset.x.shape[0]
        node_id = np.arange(idx_max).reshape(-1, 1)
        dataset.y = np.concatenate([node_id, dataset.y.reshape(-1, 1)], axis=1)
    elif name == "AmazonSparse":
        dataset = AmazonSparseDataset(root_dir)
        idx_max = dataset.idx_max
    elif name == "ogbn-mag":
        dataset = OGBNmagDataset(root_dir)
        idx_max = dataset.idx_max
    else:
        raise NotImplementedError
    return dataset, int(idx_max)
