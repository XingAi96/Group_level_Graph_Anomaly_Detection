from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import coalesce_edge_index
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx, sort_edge_index
import networkx as nx
import numpy as np
import torch
import time

class SubDecreasing(Augmentor):
    def __init__(self):
        super(SubDecreasing, self).__init__()


    def cycle_augment(self, x, edge_index, edge_weights, batch, cycle_edges_list):

        edge_index_copy = edge_index.clone().cpu().numpy().T
        edge_index_indicate = edge_index.clone().cpu().numpy().T
        cycle_edges_list = cycle_edges_list

        idx_list = []
        for idx in range(edge_index_indicate.shape[0]):
            edge = (edge_index_indicate[idx][0], edge_index_indicate[idx][1])
            if edge in cycle_edges_list:
                idx_list.append(idx)

        edge_index_copy = np.delete(edge_index_copy, np.array(idx_list), 0)
        edge_index_copy = torch.from_numpy(edge_index_copy.T).to(edge_index.device)

        return x, edge_index_copy, edge_weights, batch


    def tree_augment(self, x, edge_index, edge_weights, batch, tree_root_list):
        edge_index_copy = edge_index.clone().cpu().numpy().T
        edge_index_indicate = edge_index.clone().cpu().numpy().T
        idx_list = []
        for idx in range(edge_index_indicate.shape[0]):
            edge = edge_index_indicate[idx]
            if edge[0] in tree_root_list or edge[1] in tree_root_list:
                idx_list.append(idx)

        edge_index_copy = np.delete(edge_index_copy, np.array(idx_list), 0)
        edge_index_copy = torch.from_numpy(edge_index_copy.T).to(edge_index.device)

        return x, edge_index_copy, edge_weights, batch


    def path_augment(self, x, edge_index, edge_weights, batch, path_middle_list):

        edge_index_copy = edge_index.clone().cpu().numpy().T
        edge_index_indicate = edge_index.clone().cpu().numpy().T
        idx_list = []
        for idx in range(edge_index_indicate.shape[0]):
            edge = edge_index_indicate[idx]
            if edge[0] in path_middle_list or edge[1] in path_middle_list:
                idx_list.append(idx)

        edge_index_copy = np.delete(edge_index_copy, np.array(idx_list), 0)
        edge_index_copy = torch.from_numpy(edge_index_copy.T).to(edge_index.device)

        return x, edge_index_copy, edge_weights, batch


    def augment(self, g: Graph, batch, cycle_edges_list = None,
    tree_root_list = None, del_edge_index = None, one_degree_list = None) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        if del_edge_index != None and len(del_edge_index) != 0:
            edge_index_copy = edge_index.clone().cpu().numpy().T
            edge_index_copy = np.delete(edge_index_copy, np.array(del_edge_index), 0)
            edge_index_copy = torch.from_numpy(edge_index_copy.T).to(edge_index.device)

            return x, edge_index_copy, edge_weights, batch

        return x, edge_index, edge_weights, batch