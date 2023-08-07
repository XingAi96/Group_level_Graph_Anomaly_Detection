from typing import *
import os
import torch
import dgl
import random
import numpy as np
from torch_geometric.utils import from_networkx, to_networkx, sort_edge_index
import networkx as nx


def split_dataset(dataset, split_mode, *args, **kwargs):
    assert split_mode in ['rand', 'ogb', 'wikics', 'preload']
    if split_mode == 'rand':
        assert 'train_ratio' in kwargs and 'test_ratio' in kwargs
        train_ratio = kwargs['train_ratio']
        test_ratio = kwargs['test_ratio']
        num_samples = dataset.x.size(0)
        train_size = int(num_samples * train_ratio)
        test_size = int(num_samples * test_ratio)
        indices = torch.randperm(num_samples)
        return {
            'train': indices[:train_size],
            'val': indices[train_size: test_size + train_size],
            'test': indices[test_size + train_size:]
        }
    elif split_mode == 'ogb':
        return dataset.get_idx_split()
    elif split_mode == 'wikics':
        assert 'split_idx' in kwargs
        split_idx = kwargs['split_idx']
        return {
            'train': dataset.train_mask[:, split_idx],
            'test': dataset.test_mask,
            'val': dataset.val_mask[:, split_idx]
        }
    elif split_mode == 'preload':
        assert 'preload_split' in kwargs
        assert kwargs['preload_split'] is not None
        train_mask, test_mask, val_mask = kwargs['preload_split']
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize(s):
    return (s.max() - s) / (s.max() - s.mean())


def build_dgl_graph(edge_index: torch.Tensor) -> dgl.DGLGraph:
    row, col = edge_index
    return dgl.graph((row, col))


def batchify_dict(dicts: List[dict], aggr_func=lambda x: x):
    res = dict()
    for d in dicts:
        for k, v in d.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)
    res = {k: aggr_func(v) for k, v in res.items()}
    return res


def sample_sub(pyg):
    #x, edge_index, edge_weights = g.unfold()

    x, edge_index = pyg.x, pyg.edge_index
    nxg = to_networkx(pyg, to_undirected=True, remove_self_loops=False)

    nodes = list(nxg.nodes)
    deg_list = sorted(nxg.degree, key=lambda x: x[1], reverse=False)

    # get the node with min degree
    min_deg_node, min_deg = deg_list[0][0], deg_list[0][1]

    if len(nodes) < 1e4 and len(nodes) > 2:

        # locate all cycles
        cycles = sorted(nx.simple_cycles(nxg.to_directed()))
        #cycles = max(cycles, key=len)
        cycle_edges = []
        for cycle in cycles:
            if len(cycle) < 3:
                continue
            edge = nxg.edges(cycle[0])
            if list(edge)[0] not in cycle_edges:
                cycle_edges.append(list(edge)[0])

        paths, trees = [], []
        one_degree_nodes = []
        if min_deg < 2:
            for node, degree in deg_list:
                if degree > 1:
                    break
                # locate trees
                trees.append(nx.bfs_tree(nxg, node))
                # locate paths
                paths.append(list(nx.dfs_preorder_nodes(nxg, node)))
                # collect one-degree nodes
                one_degree_nodes.append(node)

        tree_root_nodes = []
        for tree in trees:
            max_degree_node = sorted(tree.degree, key=lambda x: x[1], reverse=True)[0]
            if max_degree_node not in tree_root_nodes:
                tree_root_nodes.append(max_degree_node[0])

        path_middle_nodes = []
        for path in paths:
            idx = int(len(path)/2)
            if path[idx] not in path_middle_nodes:
                path_middle_nodes.append(path[idx])

        edge_index_indicate = edge_index.clone().cpu().numpy().T
        del_edge_index = []
        for idx in range(edge_index_indicate.shape[0]):
            edge = (edge_index_indicate[idx][0], edge_index_indicate[idx][1])
            if edge in cycle_edges or \
                    edge[0] in tree_root_nodes or edge[1] in tree_root_nodes or\
                edge[0] in path_middle_nodes or edge[1] in path_middle_nodes:
                del_edge_index.append(idx)

        return cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes

    elif len(nodes) == 2:
        return [], [], [list(nx.dfs_preorder_nodes(G, min_deg_node))], [deg_list[0][0], deg_list[1][0]]

    else:
        return [], [], [], []