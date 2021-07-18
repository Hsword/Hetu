import hetu
import graphmix
import numpy as np
from tqdm import tqdm


def padding(graph, target_num_nodes):
    assert graph.num_nodes <= target_num_nodes
    graph.convert2coo()
    new_graph = graphmix.Graph(graph.edge_index, target_num_nodes)
    new_graph.tag = graph.tag
    new_graph.type = graph.type
    extra = target_num_nodes - graph.num_nodes
    new_graph.i_feat = np.concatenate(
        [graph.i_feat, np.tile(graph.i_feat[0], [extra, 1])])
    new_graph.f_feat = np.concatenate(
        [graph.f_feat, np.tile(graph.f_feat[0], [extra, 1])])
    if graph.extra.size:
        new_graph.extra = np.concatenate([graph.extra, np.zeros([extra, 1])])
    return new_graph


def prepare_data(ngraph):
    cli = graphmix.Client()
    graphs = []
    for i in tqdm(range(ngraph)):
        query = cli.pull_graph()
        graph = cli.wait(query)
        graphs.append(graph)
    max_num_nodes = 0
    for i in range(ngraph):
        max_num_nodes = max(max_num_nodes, graphs[i].num_nodes)
    for i in range(ngraph):
        graphs[i] = padding(graphs[i], max_num_nodes)
    return graphs


def get_norm_adj(graph, device, use_original_gcn_norm=False):
    norm = graph.gcn_norm(use_original_gcn_norm)
    mp_mat = hetu.ndarray.sparse_array(
        values=norm,
        indices=(graph.edge_index[1], graph.edge_index[0]),
        shape=(graph.num_nodes, graph.num_nodes),
        ctx=device
    )
    return mp_mat
