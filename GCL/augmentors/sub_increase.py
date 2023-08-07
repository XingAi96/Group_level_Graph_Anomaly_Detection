from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import coalesce_edge_index
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx, sort_edge_index
import networkx as nx
import numpy as np
import torch
import time

class SubIncreasing(Augmentor):
    def __init__(self):
        super(SubIncreasing, self).__init__()


    def augment(self, g: Graph, batch, cycle_edges_list = None,
    tree_root_list = None, del_edge_index = None, one_degree_list = None) -> Graph:
        x, edge_index, edge_weights = g.unfold()
        node_num = edge_index.max().item() + 1
        x_mean = torch.mean(x, dim=0).unsqueeze(dim=0)
        batch_copy = batch.clone().cpu().numpy()

        new_edges, new_batch = [], []
        new_node_num = 0
        if one_degree_list != None:
            new_node_num += len(one_degree_list)
            new_batch += list(batch_copy[one_degree_list])
        if tree_root_list != None:
            new_node_num += len(tree_root_list)
            new_batch += list(batch_copy[tree_root_list])
        if new_node_num != 0:
            new_nodes = list(range(node_num, node_num+new_node_num))
            node_index = one_degree_list + tree_root_list
            new_edges = np.array([node_index+new_nodes, new_nodes+node_index])
            new_batch = batch_copy[node_index]
            new_x = x_mean.repeat(new_node_num, 1)
            x = torch.cat([x, new_x], dim=0)

        new_node_index = node_num+new_node_num
        new_edges = list(new_edges.T)
        new_batch = list(new_batch)
        if cycle_edges_list != None:
            for edges in cycle_edges_list:
                node1, node2 = edges[0], edges[1]
                x = torch.cat([x, x_mean], dim=0)
                new_batch.append(batch_copy[node1])
                new_edges.append((node1, new_node_index))
                new_edges.append((node2, new_node_index))
                new_node_index += 1

        if len(new_batch) != 0:
            new_batch = torch.from_numpy(np.array(new_batch)).T.to(batch.device)
            batch = torch.cat([batch, new_batch])
            new_edges = torch.from_numpy(np.array(new_edges)).T.to(edge_index.device)
            edge_index = torch.cat([edge_index, new_edges], dim=1)

        return x, edge_index, edge_weights, batch.type(torch.int64)

    def augment_old(self, g: Graph, batch, cycle_edges_list = None,
    tree_root_list = None, del_edge_index = None, one_degree_list = None) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        if one_degree_list != None:
            x, edge_index, edge_weights, batch = self.path_augment(x, edge_index, edge_weights, batch, one_degree_list)
        if cycle_edges_list != None:
           x, edge_index, edge_weights, batch = self.cycle_augment(x, edge_index, edge_weights, batch, cycle_edges_list)
        if tree_root_list != None:
           x, edge_index, edge_weights, batch = self.tree_augment(x, edge_index, edge_weights, batch, tree_root_list)

        return x, edge_index, edge_weights, batch.type(torch.int64)
