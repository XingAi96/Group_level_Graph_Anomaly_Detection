import os
import csv
import json
import time
import math
import torch
import random
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import networkx as nx
from tqdm import tqdm
import scipy
import scipy.io as sio
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch_geometric.utils import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import laplacian
from sklearn.preprocessing import normalize, StandardScaler
import torch_geometric.transforms as T


class GraphProcessor:

    def relabel_old(self, data, comp_set):
        i = 0
        y = []
        comp_size = 0
        is_mix, complete_ratio, redun_ratio = [], {}, {}
        batch = torch.zeros(data.num_nodes, dtype=torch.long)
        for comp in comp_set:
            comp = list(comp)
            comp_size += len(comp)

            if data.y.shape[0] == data.num_nodes:
                is_ano = [node in data.y for node in comp]
            else:
                is_ano = [data.y.to(torch.bool)[data.batch[v]] for v in comp]
            # y.append(1 if True in is_ano else 0)
            y.append(0 if False in is_ano else 1)

            batch[comp] = i
            i += 1

            # measurement
            graph_idx = []
            for v in comp:
                graph_idx.append(list(data.batch[v].cpu().numpy().reshape(-1))[0])
            graph_idx = list(set(graph_idx))
            if len(graph_idx) > 1:
                is_mix.append(1)
            else:
                is_mix.append(0)

            node_set_list, label_list = [], []
            for gi in graph_idx:
                gi = int(gi)
                indicator = torch.where(data.batch == gi, 1, 0)
                node_set = set(list(torch.nonzero(indicator).cpu().squeeze().numpy()))
                node_set_list.append(node_set)
                label_list.append(list(data.y[gi].cpu().numpy().reshape(-1))[0])
                if label_list[-1] == 1:
                    lost_node, redun_node = node_set.difference(set(comp)), set(comp).difference(node_set)
                    if gi not in list(complete_ratio.keys()) or 1 - (len(lost_node) / len(node_set)) > complete_ratio[
                        gi]:
                        complete_ratio[gi] = 1. - (len(lost_node) / len(node_set))
                    if gi not in list(redun_ratio.keys()) or 1 - (len(redun_node) / len(comp)) > redun_ratio[gi]:
                        redun_ratio[gi] = 1 - (len(redun_node) / len(comp))
        comp_size /= len(comp_set)

        ground_truth = list(data.y.cpu().numpy())
        com_ratio, red_ratio = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        for gid in range(len(ground_truth)):
            if ground_truth[gid] != 1:
                continue
            if gid not in list(complete_ratio.keys()):
                complete_ratio[gid] = 0.
            if complete_ratio[gid] <= 0.1:
                com_ratio[0] += 1
            elif complete_ratio[gid] <= 0.4:
                com_ratio[1] += 1
            elif complete_ratio[gid] <= 0.7:
                com_ratio[2] += 1
            elif complete_ratio[gid] <= 0.999:
                com_ratio[3] += 1
            else:
                com_ratio[4] += 1

            if gid not in list(redun_ratio.keys()):
                redun_ratio[gid] = 0.
            if redun_ratio[gid] <= 0.1:
                red_ratio[0] += 1
            elif redun_ratio[gid] <= 0.4:
                red_ratio[1] += 1
            elif redun_ratio[gid] <= 0.7:
                red_ratio[2] += 1
            elif redun_ratio[gid] <= 0.999:
                red_ratio[3] += 1
            else:
                red_ratio[4] += 1

        ratio = [[c / sum(com_ratio) for c in com_ratio],
                 [r / sum(red_ratio) for r in red_ratio]]
        return y, batch, is_mix, ratio, comp_size

    def sample_sub(self, data, mean_error_list, threshold):

        nxg = data.G
        nodes = np.array(nxg.nodes)
        center_nodes = nodes[mean_error_list > threshold]

        # locate candidate subgraphs
        subdata = []
        subgraphs = []
        tree_root_nodes, path_middle_nodes, cycle_edges = [], [], []
        for i in range(len(center_nodes)):

            # find circuits
            try:
                edges = nx.find_cycle(nxg, center_nodes[i])
                edges = set(edges)
                circuit = []
                for edge in edges:
                    if edge[0] not in circuit:
                        circuit.append(edge[0])
                    if edge[1] not in circuit:
                        circuit.append(edge[1])
                if set(circuit) not in subgraphs:
                    subgraphs.append(set(circuit))
                    cycle_edges.append(circuit[int(len(circuit)/2)])
            except:
                # there is no circuit
                pass

            # find trees
            try:
                tree = nx.bfs_tree(nxg, center_nodes[i], 1)
                tree_nodes = set(tree.nodes)
                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                    subgraphs.append(tree_nodes)
                    max_degree_node = sorted(tree.degree, key=lambda x: x[1], reverse=True)[0]
                    tree_root_nodes.append(max_degree_node[0])
            except:
                # there is no tree
                pass

            # find paths between anomaly nodes
            for j in range(i + 1, len(center_nodes)):
                try:
                    dis = nx.shortest_path_length(nxg, center_nodes[i], center_nodes[j])
                    if dis > 2e2:
                        continue
                    '''
                    # Extracting breadth-first/depth-first graph between i and j to faster the path search
                    dfsg1 = nx.dfs_tree(nxg, center_nodes[i], dis)
                    dfsg2 = nx.dfs_tree(nxg, center_nodes[j], dis)
                    try:
                        dfsg = nx.intersection(dfsg1, dfsg2)
                    except:
                        continue
                    path_list = list(nx.shortest_simple_paths(dfsg, center_nodes[i], center_nodes[j]))
                    '''
                    path_list = nx.shortest_paths(nxg, center_nodes[i], center_nodes[j])
                    # travel all paths
                    for path in path_list:
                        for node in path:
                            if node == center_nodes[i] or node == center_nodes[j]:
                                continue

                            path_i, path_j = [], []
                            # calculate distance to node i/j
                            dis_i = nx.shortest_path_length(nxg, center_nodes[i], node)
                            dis_j = nx.shortest_path_length(nxg, center_nodes[j], node)
                            if node in center_nodes and dis_i < 5:
                                bfs_tree_i = nx.bfs_tree(nxg, center_nodes[i], dis_i)
                                tree_nodes = set(bfs_tree_i.nodes)
                                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                                    subgraphs.append(tree_nodes)
                                    max_degree_node = sorted(bfs_tree_i.degree, key=lambda x: x[1], reverse=True)[0]
                                    tree_root_nodes.append(max_degree_node[0])
                                path_i = list(nx.shortest_simple_paths(bfs_tree_i, center_nodes[i], node))
                            if node in center_nodes and dis_j < 5:
                                bfs_tree_j = nx.bfs_tree(nxg, center_nodes[j], dis_j)
                                tree_nodes = set(bfs_tree_j.nodes)
                                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                                    subgraphs.append(tree_nodes)
                                    max_degree_node = sorted(bfs_tree_j.degree, key=lambda x: x[1], reverse=True)[0]
                                    tree_root_nodes.append(max_degree_node[0])
                                path_j = list(nx.shortest_simple_paths(bfs_tree_j, center_nodes[j], node))

                            middle_paths = path_i + path_j
                            for mp in middle_paths:
                                if set(mp) not in subgraphs:
                                    path_middle_nodes.append(mp[int(len(mp)/2)])
                                    subgraphs.append(set(mp))
                except:
                    # there is no path
                    pass

        # generate residual data
        sub_index, sub_size = 0, 0
        sub_label, batch = [], torch.zeros(data.num_nodes, dtype=torch.long)
        is_mix = []
        for sub in subgraphs:
            subdata += list(sub)
            sub_size += len(sub)
            is_ano = [data.y.to(torch.bool)[data.batch[v]] for v in sub]
            # sub_label.append(1 if True in is_ano else 0)
            sub_label.append(0 if False in is_ano else 1)
            batch[list(sub)] = sub_index
            sub_index += 1

            # measurement
            graph_idx = []
            for v in sub:
                graph_idx.append(list(data.batch[v].cpu().numpy().reshape(-1))[0])
            graph_idx = list(set(graph_idx))
            if len(graph_idx) > 1:
                is_mix.append(1)
            else:
                is_mix.append(0)
            node_set_list, label_list = [], []
            for gi in graph_idx:
                gi = int(gi)
                indicator = torch.where(data.batch == gi, 1, 0)
                try:
                    node_set = set(list(torch.nonzero(indicator).cpu().squeeze().numpy()))
                except:
                    indicator = indicator.cpu().squeeze().numpy()
                    indicator = np.nonzero(indicator)
                    indicator = list(indicator[0])
                    node_set = set(indicator)
                node_set_list.append(node_set)
                label_list.append(list(data.y[gi].cpu().numpy().reshape(-1))[0])
                if label_list[-1] == 1:
                    lost_node, redun_node = node_set.difference(set(sub)), set(sub).difference(node_set)
        sub_size /= len(subgraphs)

        subdata = list(set(subdata))
        residual_edge_index, _ = subgraph(
            torch.arange(data.num_nodes)[subdata].to(data.x.device),
            data.edge_index,
            relabel_nodes=True
        )
        residal_data = Data(
            # x=torch.tensor(mean_error_list[mean_error_list > threshold]).view(-1, 1),
            x=data.x[subdata],
            edge_index=residual_edge_index,
            batch = batch[subdata],
            y = torch.from_numpy(np.array(sub_label))
        )
        residal_data.to(data.x.device)
        return subgraphs, residal_data, sub_size

    def sample_aug_sub(self, pyg):
        # x, edge_index, edge_weights = g.unfold()

        x, edge_index = pyg.x, pyg.edge_index
        nxg = to_networkx(pyg, to_undirected=True, remove_self_loops=False)

        nodes = list(nxg.nodes)
        deg_list = sorted(nxg.degree, key=lambda x: x[1], reverse=False)

        # get the node with min degree
        min_deg_node, min_deg = deg_list[0][0], deg_list[0][1]

        if len(nodes) > 2:

            paths, trees, cycles = [], [], []
            one_degree_nodes = []
            for node, degree in deg_list:
                if degree > 1:
                    # locate cycles
                    try:
                        cyc_edges = nx.find_cycle(nxg, node)
                        if set(cyc_edges) not in cycles:
                            cycles.append(set(cyc_edges))
                    except:
                        pass
                else:
                    # locate trees
                    trees.append(nx.bfs_tree(nxg, node))
                    # locate paths
                    paths.append(list(nx.dfs_preorder_nodes(nxg, node)))
                    # collect one-degree nodes
                    one_degree_nodes.append(node)

            cycle_edges = []
            for cycle in cycles:
                if len(cycle) < 3:
                    continue
                for idx in range(len(cycle)):
                    edge = list(list(cycle)[idx])
                    if edge not in cycle_edges:
                        cycle_edges.append(edge)
                        break

            tree_root_nodes = []
            for tree in trees:
                max_degree_node = sorted(tree.degree, key=lambda x: x[1], reverse=True)[0]
                if max_degree_node not in tree_root_nodes:
                    tree_root_nodes.append(max_degree_node[0])

            path_middle_nodes = []
            for path in paths:
                idx = int(len(path) / 2)
                if path[idx] not in path_middle_nodes:
                    path_middle_nodes.append(path[idx])

            edge_index_indicate = edge_index.clone().cpu().numpy().T
            del_edge_index = []
            for idx in range(edge_index_indicate.shape[0]):
                edge = (edge_index_indicate[idx][0], edge_index_indicate[idx][1])
                if edge in cycle_edges or \
                        edge[0] in tree_root_nodes or edge[1] in tree_root_nodes or \
                        edge[0] in path_middle_nodes or edge[1] in path_middle_nodes:
                    del_edge_index.append(idx)

            return cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes

        # elif len(nodes) == 2:
        #    return [], [], [list(nx.dfs_preorder_nodes(G, min_deg_node))], [deg_list[0][0], deg_list[1][0]]

        else:
            return [], [], [], []

    def subgraph_ano_injection(self, G: Data, ano_sub: dict):
        '''
        subgraph anomaly injection.
        '''
        mu, sigma = 1, 0.1
        path_count, tree_count, cir_count = 0, 0, 0
        ano_sub_sizes, ano_sub_type = [], []
        for type in list(ano_sub.keys()):
            if type == 'path':
                path_count = ano_sub['path']
                path_sizes = np.random.randint(3, 10, path_count)
                ano_sub_sizes += path_sizes.tolist()
                [ano_sub_type.append('path') for i in range(path_count)]

            if type == 'tree':
                tree_count = ano_sub['tree']
                tree_sizes = np.random.randint(3, 10, tree_count)
                ano_sub_sizes += tree_sizes.tolist()
                [ano_sub_type.append('tree') for i in range(tree_count)]

            if type == 'circuit':
                cir_count = ano_sub['circuit']
                cir_sizes = np.random.randint(3, 10, cir_count)
                ano_sub_sizes += cir_sizes.tolist()
                [ano_sub_type.append('circuit') for i in range(cir_count)]
        tol_count = path_count + tree_count + cir_count
        tol_size = np.sum(path_sizes) + np.sum(tree_sizes) + np.sum(cir_sizes)

        # prepare new anomaly nodes, including node lables and node features
        ano_node_index = np.array(range(G.num_nodes, G.num_nodes + tol_size))
        np.random.shuffle(ano_node_index)

        # sample anchor nodes
        anchor_index, anchor_neighbors = [], []
        start_index = 0
        nxg = to_networkx(G)
        batch = np.zeros((G.num_nodes + tol_size,))
        ano_feature = np.ones((tol_size, G.num_features))
        x = G.x.numpy()
        edge_index = G.edge_index.numpy().T.tolist()
        for idx in range(tol_count):
            while True:
                cand_node = np.random.choice(G.num_nodes, 1, replace=False).tolist()[0]
                if cand_node not in anchor_index and cand_node not in anchor_neighbors:
                    break
            anchor_index.append(cand_node)
            neis = list(nxg.neighbors(cand_node))
            anchor_neighbors += neis

            # add new nodes as anomaly nodes
            x[cand_node] = np.random.normal(mu, sigma, x.shape[1])
            features = [x[cand_node]]
            size = ano_sub_sizes[idx]
            while True:
                cand_neis = np.random.choice(G.num_nodes, size, replace=False).tolist()
                for cn in cand_neis:
                    if cn not in neis:
                        size -= 1
                        x[cn] = np.random.normal(mu, sigma, x.shape[1])
                        features.append(x[cn])
                        if size == 0:
                            break
                if size == 0:
                    break

            # link anchor node and new nodes according to the sub_type
            size = ano_sub_sizes[idx]
            sub_nodes = ano_node_index[start_index: start_index + size]
            for i in range(size):
                ano_feature[sub_nodes[i] - G.num_nodes] = features[i]

            if ano_sub_type[idx] == 'path':
                for i in range(size - 1):
                    edge_index.append((sub_nodes[i], sub_nodes[i + 1]))
                    edge_index.append((sub_nodes[i + 1], sub_nodes[i]))

            elif ano_sub_type[idx] == 'tree':
                parent_node = np.random.choice(sub_nodes)
                for i in range(size):
                    if sub_nodes[i] == parent_node:
                        continue
                    edge_index.append((parent_node, sub_nodes[i]))
                    edge_index.append((sub_nodes[i], parent_node))

            elif ano_sub_type[idx] == 'circuit':
                for i in range(size - 1):
                    edge_index.append((sub_nodes[i], sub_nodes[i + 1]))
                    edge_index.append((sub_nodes[i + 1], sub_nodes[i]))
                edge_index.append((sub_nodes[0], sub_nodes[-1]))
                edge_index.append((sub_nodes[-1], sub_nodes[0]))

            batch[sub_nodes] = idx + 1
            start_index += size

        # expand the origin graph
        ano_feature = torch.tensor(ano_feature, dtype=torch.float32)
        G.x = torch.cat([G.x, ano_feature], dim=0)
        G.node_label = torch.cat([G.y, torch.from_numpy(np.ones(tol_size))], dim=0)
        G.y = np.ones((tol_count+1,))
        G.y[0] = 0
        G.y = torch.from_numpy(G.y)
        G.batch = torch.tensor(batch, dtype=torch.int64)
        G.edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)

        return G

    def sample_sub_old(self, data, mean_error_list, threshold):

        nxg = data.G
        nodes = np.array(nxg.nodes)
        center_nodes = nodes[mean_error_list > threshold]

        paths = []
        subgraphs = []
        for i in range(len(center_nodes)):

            # find circuits
            try:
                edges = nx.find_cycle(nxg, center_nodes[i])
                edges = set(edges)
                circuit = []
                for edge in edges:
                    if edge[0] not in circuit:
                        circuit.append(edge[0])
                    if edge[1] not in circuit:
                        circuit.append(edge[1])
                if set(circuit) not in subgraphs:
                    subgraphs.append(set(circuit))
            except:
                # there is no circuit
                pass

            # find trees
            try:
                tree_nodes = set(nx.bfs_tree(nxg, center_nodes[i], 1).nodes)
                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                    subgraphs.append(tree_nodes)
            except:
                # there is no tree
                pass

            # find paths between anomaly nodes
            for j in range(i + 1, len(center_nodes)):
                try:
                    dis = nx.shortest_path_length(nxg, center_nodes[i], center_nodes[j])
                    if dis > 2e2:
                        continue
                    '''
                    # Extracting breadth-first/depth-first graph between i and j to faster the path search
                    dfsg1 = nx.dfs_tree(nxg, center_nodes[i], dis)
                    dfsg2 = nx.dfs_tree(nxg, center_nodes[j], dis)
                    try:
                        dfsg = nx.intersection(dfsg1, dfsg2)
                    except:
                        continue
                    path_list = list(nx.shortest_simple_paths(dfsg, center_nodes[i], center_nodes[j]))
                    '''
                    path_list = nx.shortest_paths(nxg, center_nodes[i], center_nodes[j])
                    # travel all paths
                    for path in path_list:
                        for node in path:
                            if node == center_nodes[i] or node == center_nodes[j]:
                                continue

                            path_i, path_j = [], []
                            # calculate distance to node i/j
                            dis_i = nx.shortest_path_length(nxg, center_nodes[i], node)
                            dis_j = nx.shortest_path_length(nxg, center_nodes[j], node)
                            if node in center_nodes and dis_i < 5:
                                bfs_tree_i = nx.bfs_tree(nxg, center_nodes[i], dis_i)
                                tree_nodes = set(bfs_tree_i.nodes)
                                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                                    subgraphs.append(tree_nodes)
                                path_i = list(nx.shortest_simple_paths(bfs_tree_i, center_nodes[i], node))
                            if node in center_nodes and dis_j < 5:
                                bfs_tree_j = nx.bfs_tree(nxg, center_nodes[j], dis_j)
                                tree_nodes = set(bfs_tree_j.nodes)
                                if len(tree_nodes) < 0.1 * len(nodes) and tree_nodes not in subgraphs:
                                    subgraphs.append(tree_nodes)
                                path_j = list(nx.shortest_simple_paths(bfs_tree_j, center_nodes[j], node))

                            middle_paths = path_i + path_j
                            for mp in middle_paths:
                                if set(mp) not in subgraphs:
                                    subgraphs.append(set(mp))
                except:
                    # there is no path
                    pass

        return subgraphs

def preprocess_data(data, type='s'):
    edge_index = data.edge_index
    A = to_dense_adj(edge_index)[0]
    A_array = A.cpu().numpy()
    G = nx.from_numpy_matrix(A_array)
    if type == '1':
        return A, G

    if type != 's':
        from sklearn import preprocessing as p

        tmp = A_array
        for i in range(int(type)-1):
            tmp = np.matmul(tmp, A_array)

        np.fill_diagonal(tmp, 0)
        min_max_scaler = p.MinMaxScaler()
        normalizedData = min_max_scaler.fit_transform(tmp)
        A = torch.tensor(normalizedData, dtype=torch.int64).to(edge_index.device)
        return A, G


    sub_graphs = []
    subgraph_nodes_list = []
    sub_graphs_adj = []
    sub_graph_edges = []
    new_adj = torch.zeros(A_array.shape[0], A_array.shape[0])

    for i in np.arange(len(A_array)):
        s_indexes = [i]#len(A_array) * [i]
        s_indexes += list(G.neighbors(i))
        sub_graphs.append(G.subgraph(s_indexes))

    for index in np.arange(len(sub_graphs)):
        subgraph_nodes_list.append(list(sub_graphs[index].nodes))
        sub_graphs_adj.append(nx.adjacency_matrix(sub_graphs[index]).toarray())
        sub_graph_edges.append(sub_graphs[index].number_of_edges())

    for node in np.arange(len(subgraph_nodes_list)):
        sub_adj = sub_graphs_adj[node]
        for neighbors in np.arange(len(subgraph_nodes_list[node])):
            index = subgraph_nodes_list[node][neighbors]
            count = torch.tensor(0).float()
            if (index == node):
                continue
            else:
                c_neighbors = set(subgraph_nodes_list[node]).intersection(subgraph_nodes_list[index])
                if index in c_neighbors:
                    nodes_list = subgraph_nodes_list[node]
                    sub_graph_index = nodes_list.index(index)
                    c_neighbors_list = list(c_neighbors)
                    for i, item1 in enumerate(nodes_list):
                        if (item1 in c_neighbors):
                            for item2 in c_neighbors_list:
                                j = nodes_list.index(item2)
                                count += sub_adj[i][j]

                new_adj[node][index] = count / 2
                new_adj[node][index] = new_adj[node][index] / (len(c_neighbors) * (len(c_neighbors) - 1))
                new_adj[node][index] = new_adj[node][index] * (len(c_neighbors) ** 2)

    weight = torch.FloatTensor(new_adj)
    weight = weight / weight.sum(1, keepdim=True)

    weight = weight + torch.FloatTensor(A_array)

    coeff = weight.sum(1, keepdim=True)
    coeff = torch.diag((coeff.T)[0])

    weight = weight + coeff
    weight = weight.detach().numpy()
    weight = np.nan_to_num(weight, nan=0)

    A = weight
    '''
    A_max, A_min = np.max(A, axis=1), np.min(A, axis=1)
    for i in np.arange(len(A_array)):
        A[i] = (A[i] - A_min[i])/(A_max[i] - A_min[i]+1)
    '''
    A = torch.tensor(A, dtype=torch.int64).to(edge_index.device)

    return A, G



class Visualizor():

    def embedding_visualization(self, x, y, dataset):
        label = []
        for i in y:
            if i == 0:
                label.append('normal group')
            else:
                label.append('anomaly group')
        y = np.array(label)

        dict_color = {'normal group': '#2ca12c', 'anomaly group': '#d62728'}
        markers = {'normal group': 'o', 'anomaly group': 'X'}

        # PCA
        X_std = StandardScaler().fit_transform(x)
        X_pca = PCA(n_components=2).fit_transform(X_std)
        X_pca = np.vstack((X_pca.T, y)).T
        new_list = []
        for row in X_pca:
            row_list = []
            row_list.append(float(row[0]))
            row_list.append(float(row[1]))
            row_list.append(row[2])
            new_list.append(row_list)
        df_pca = pd.DataFrame(new_list, columns=['Dim1', 'Dim2', 'class'], dtype=float)
        df_pca.head()

        # t-SNE
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(x)
        X_tsne = np.vstack((X_tsne.T, y)).T
        new_list = []
        for row in X_tsne:
            row_list = []
            row_list.append(float(row[0]))
            row_list.append(float(row[1]))
            row_list.append(row[2])
            new_list.append(row_list)
        df_tsne = pd.DataFrame(new_list, columns=['Dim1', 'Dim2', 'class'], dtype=float)
        df_tsne.head()

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_pca, x='Dim1', y='Dim2', hue='class', style='class', s=150, markers=markers,
                        palette=dict_color, legend=False)
        plt.savefig('./visualization/' + dataset + '_PCA_' + str(time.time()) + '.pdf', format="pdf",
                    bbox_inches="tight")

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='class', style='class', s=150, markers=markers,
                        palette=dict_color, legend=False)
        plt.savefig('./visualization/' + dataset + '_TSNE_' + str(time.time()) + '.pdf', format="pdf",
                    bbox_inches="tight")

    def augmentation_visualization(self,dataname='eth'):

        if dataname == 'AMLPublic':
            F1 = np.array(
                [[0.8847, 0.9008, 0.8855, 0.8825, 0.8882],
                 [0.8531, 0.8519, 0.8678, 0.8683, 0.8672],
                 [0.8545, 0.8876, 0.8547, 0.8736, 0.8713],
                 [0.8562, 0.8858, 0.8720, 0.8528, 0.8714],
                 [0.8535, 0.8845, 0.8536, 0.8728, 0.8733],
                 ]
            )
            AUC = np.array(
                [[0.698, 0.734, 0.694, 0.710, 0.699],
                 [0.701, 0.683, 0.680, 0.688, 0.667],
                 [0.670, 0.715, 0.70, 0.709, 0.667],
                 [0.678, 0.751, 0.716, 0.719, 0.703],
                 [0.699, 0.702, 0.695, 0.70, 0.709],
                 ]
            )

        if dataname == 'eth':
            F1 = np.array(
                [[0.698, 0.734, 0.694, 0.710, 0.699],
                 [0.701, 0.683, 0.680, 0.688, 0.667],
                 [0.670, 0.715, 0.70, 0.709, 0.667],
                 [0.678, 0.751, 0.716, 0.719, 0.703],
                 [0.699, 0.702, 0.695, 0.70, 0.709],
                 ]
            )
            AUC = np.array(
                [[0.6953, 0.8592, 0.8484, 0.6585, 0.8529],
                 [0.8275, 0.8050, 0.8195, 0.6618, 0.8572],
                 [0.8292, 0.8423, 0.8233, 0.8068, 0.8435],
                 [0.8328, 0.8517, 0.8451, 0.8467, 0.8087],
                 [0.8186, 0.8572, 0.8432, 0.8085, 0.7959],
                 ]
            )

        if dataname == 'simML':
            F1 = np.array(
                [[0.738, 0.761, 0.711, 0.749, 0.765],
                 [0.725, 0.706, 0.726, 0.718, 0.716],
                 [0.706, 0.773, 0.688, 0.678, 0.714],
                 [0.717, 0.762, 0.680, 0.712, 0.721],
                 [0.717, 0.730, 0.750, 0.723, 0.706],
                 ]
            )
            AUC = np.array(
                [[0.8070, 0.8358, 0.7917, 0.8051, 0.8408],
                 [0.8102, 0.8038, 0.8306, 0.8151, 0.7838],
                 [0.8015, 0.8533, 0.7729, 0.7276, 0.7908],
                 [0.8111, 0.8012, 0.7742, 0.7799, 0.8033],
                 [0.8040, 0.8424, 0.8426, 0.8099, 0.7845],
                 ]
            )

        if dataname == 'Cora':
            F1 = np.array(
                [[0.716, 0.75, 0.727, 0.707, 0.727],
                 [0.733, 0.717, 0.72, 0.723, 0.706],
                 [0.719, 0.71, 0.717, 0.702, 0.718],
                 [0.723, 0.723, 0.722, 0.719, 0.725],
                 [0.721, 0.711, 0.717, 0.708, 0.71],
                 ]
            )

        if dataname == 'Citeseer':
            F1 = np.array(
                [[0.847, 0.853, 0.839, 0.84, 0.839],
                 [0.827, 0.825, 0.826, 0.836, 0.82],
                 [0.837, 0.821, 0.828, 0.817, 0.826],
                 [0.823, 0.85, 0.843, 0.817, 0.835],
                 [0.84, 0.819, 0.831, 0.838, 0.84],
                 ]
            )

        data = F1
        data = pd.DataFrame(data)
        data.columns = ['PBA', 'PPA', 'ND', 'ER', 'FM']
        data.index = ['PBA', 'PPA', 'ND', 'ER', 'FM']

        # sns.set_context({"figure.figsize": (8, 8)})
        ax = plt.figure(figsize=(18, 18))
        ax = sns.heatmap(data=data, cmap='OrRd', linewidths=3., fmt='.3f', cbar_kws={'shrink': 0.83},
                         annot=True, annot_kws={'size': 42, 'rotation': -60., 'weight': 'bold'}, square=True)
        '''
        label_y = ['SD', 'SI','ND','ER','FM']
        plt.setp(label_y, rotation=45, horizontalalignment='right')
        label_x = ['SD', 'SI','ND','ER','FM']
        plt.setp(label_x, rotation=45, horizontalalignment='right')
        '''
        ax.xaxis.set_tick_params(labelsize=40)
        ax.yaxis.set_tick_params(labelsize=40)
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=36)
        plt.savefig("Aug_ablation_" + dataname + ".pdf", format="pdf", bbox_inches="tight")

        return

    def group_size_visualization(self):

        group_size = np.array(
            [[1.85, 2.15, 2.28, 1.70, 2.24],
             [1.82, 2.15, 2.82, 1.73, 2.26],
             [1.91, 2.13, 2.77, 1.73, 2.2],
             [1.15, 1.18, 1.23, 1.75, 2.13],
             [1.57, 3.31, 1.68, 2.97, 2.46],
             [2.58, 4.11, 13.80, 4.44, 4.1],
             [3.52, 7.23, 19.05, 6.32, 6.18]
             ]
        )
        '''
        data = np.array(
            [['', 'AMLpublic', 'Ethereum_TSGN','simML','Cora','Citeseer'],
                ['Dominant', 1.85, 2.15, 2.28, 1.70, 2.24],
            ['DeepAE', 1.82, 2.15, 2.82, 1.73, 2.26],
             ['ComGA', 1.91, 2.13, 2.77, 1.73, 2.2],
             ['DeepFD', 1.15, 1.18, 1.23, 1.75, 2.13],
             ['AS-GAE', 1.57, 3.31, 1.68, 2.97, 2.46],
             ['Ours', 2.58, 4.11, 13.80, 4.44, 4.1],
             ['Ground Truth', 3.52, 7.23, 19.05, 6.32, 6.18]
            ]
        )
        '''

        datasets = ['AMLpublic', 'Ethereum_TSGN', 'simML', 'Cora_group', 'Citeseer_group']
        baselines = ['Dominant', 'DeepAE', 'ComGA', 'DeepFD', 'AS-GAE', 'Ours', 'Ground Truth']

        data = []
        ds_idx = 0
        for ds in datasets:
            bs_idx = 0
            for bs in baselines:
                row = [ds, bs, group_size[bs_idx][ds_idx]]
                data.append(row)
                bs_idx += 1
            ds_idx += 1
        data = pd.DataFrame(data, columns=['Datasets', 'Models', 'Group size'], dtype=float)

        ax = plt.figure(figsize=(24, 14))
        sns.set_style('whitegrid')
        ax = sns.barplot(data, x='Group size', y='Datasets', orient='h', hue='Models', width=.9)

        ax.xaxis.set_tick_params(labelsize=38)
        ax.yaxis.set_tick_params(labelsize=32, rotation=45)

        plt.legend(title='Models', fontsize=30, title_fontsize=30)
        plt.xlabel('Group size', fontsize=28);
        # plt.ylabel('Group_size', fontsize=32);
        plt.savefig("group_size.pdf", format="pdf")

        return

    def group_pattern_visualization(self, dataname, data):
        # ground truth
        groups = []
        group_patterns = []
        for gid in range(data.y.shape[0]):
            if data.y[gid] == 1:
                patterns = []
                indicator = torch.where(data.batch == gid, 1, 0)
                nodes = list(torch.nonzero(indicator).cpu().squeeze().numpy())
                edges, _ = subgraph(torch.arange(data.num_nodes)[nodes],data.edge_index.cpu(),relabel_nodes=True)
                group = Data(x=data.x[nodes], edge_index=edges)
                group = to_networkx(group)
                patterns.append(1)
                cycles = list(nx.cycle_basis(group.to_undirected()))
                if len(cycles) == 0:
                    patterns[0] = 0
                patterns.append(len(group) - 1 == group.number_of_edges())
                group_patterns.append(patterns)

                pos = nx.drawing.kamada_kawai_layout(group)
                # pos = nx.shell_layout(group)
                # pos = nx.spectral_layout(G)
                # pos = nx.spring_layout(G)
                # pos = nx.random_layout(G)
                # pos = nx.circular_layout(G)

                nx.draw_networkx(group.to_undirected(), pos=pos, arrows=True, arrowsize=1, with_labels=False, width=1.2,
                                 node_size=80, node_color='blue', edge_color='gray')
                plt.axis('off')
                plt.savefig('visualization/'+dataname+'_'+str(gid)+".pdf", format="pdf", bbox_inches="tight")
                plt.clf()

        return group_patterns


def CR_calculator(data, candi_groups, y_score, y_pre):
    # ground truth
    groups = []
    group_size = 0
    for gid in range(data.y.shape[0]):
        if data.y[gid] == 1:
            indicator = torch.where(data.batch == gid, 1, 0)
            node_set = set(list(torch.nonzero(indicator).cpu().squeeze().numpy()))
            groups.append(node_set)
            group_size += len(node_set)
    group_size /= len(groups)
    # CR value
    cr_list = []
    for idx in range(len(candi_groups)):
        cg = candi_groups[idx]
        if y_pre[idx] == 0:
            continue
        max_cr = 0
        for group in groups:
            inter = len(group.intersection(cg))
            if len(cg) == 0:
                tmp = 0
            else:
                tmp = 0.5 * (inter/len(group) + inter/len(cg))
            if tmp > max_cr:
                max_cr = tmp
        cr_list.append(max_cr)
    cr1 = np.array(cr_list).mean()

    cr_list = []
    for group in groups:
        group_size += len(group)
        max_cr = 0
        for cg in candi_groups:
            inter = len(group.intersection(cg))
            tmp = 0.5 * (inter/len(group) + inter/len(cg))
            if tmp > max_cr:
                max_cr = tmp
        cr_list.append(max_cr)
    cr2 = np.array(cr_list).mean()

    cr = cr2
    return cr


class DataProcessor:

    def __ini__(self):
        self.ano_sub_count = {'tree': 10, 'path': 8, 'circuit': 3}
        self.ano_sub_size = {'path': [3, 10], 'tree': [3, 10], 'circuit': [3, 10]}
        return

    def preprocess_Chartalist(self):
        '''
        subgraph anomaly injection.
        '''
        dataset_path = r'./data/ML/Bitcoin.pt'

        G = torch.load(dataset_path)
        x = G.x.cpu().numpy()
        path_count, tree_count, cir_count = 0, 0, 0
        ano_sub_sizes, ano_sub_type, batch = [], [], []
        for type in list(self.ano_sub.keys()):
            if type == 'path':
                path_count = self.ano_sub['path']
                path_sizes = np.random.randint(self.ano_sub_size['path'][0], self.ano_sub_size['path'][1], path_count)
                ano_sub_sizes += path_sizes.tolist()
                [ano_sub_type.append('path') for i in range(path_count)]

            if type == 'tree':
                tree_count = self.ano_sub['tree']
                tree_sizes = np.random.randint(self.ano_sub_size['tree'][0], self.ano_sub_size['tree'][1], tree_count)
                ano_sub_sizes += tree_sizes.tolist()
                [ano_sub_type.append('tree') for i in range(tree_count)]

            if type == 'circuit':
                cir_count = self.ano_sub['circuit']
                cir_sizes = np.random.randint(self.ano_sub_size['circuit'][0], self.ano_sub_size['circuit'][1], cir_count)
                ano_sub_sizes += cir_sizes.tolist()
                [ano_sub_type.append('circuit') for i in range(cir_count)]
        tol_count = path_count + tree_count + cir_count
        tol_size = np.sum(path_sizes) + np.sum(tree_sizes) + np.sum(cir_sizes)

        # prepare new anomaly nodes, including node lables and node features
        anchor_index = np.random.choice(G.num_nodes, int(tol_count * 1.3), replace=False).tolist()
        ano_node_index = np.array(range(G.num_nodes, G.num_nodes + tol_size))
        np.random.shuffle(ano_node_index)
        # ano_feature = (torch.randn(G.x.size()) * torch.std(G.x) + torch.mean(G.x))[:tol_size]
        ano_feature = np.random.choice(G.x.size()[0], tol_size, replace=False)
        ano_feature = G.x[ano_feature]
        ano_feature += torch.tensor(np.random.normal(-1, 1, ano_feature.shape), dtype=torch.float64)

        # expand the origin graph
        G.x = torch.cat([G.x, ano_feature], dim=0).type(torch.float32)
        G.y = torch.cat([G.y, torch.from_numpy(np.ones(tol_size))], dim=0)
        y = np.ones((tol_count + 1))
        y[0] = 0

        # prepare subgraphs
        batch = np.zeros((G.x.shape[0]))
        start_index = 0
        sub_indicator = {}
        edge_index = G.edge_index.numpy().T.tolist()
        for sub_index in range(tol_count):
            size = ano_sub_sizes[sub_index]
            sub_nodes = ano_node_index[start_index: start_index + size]
            batch[sub_nodes] = sub_index + 1

            if ano_sub_type[sub_index] == 'path':
                for i in range(size - 1):
                    edge_index.append((sub_nodes[i], sub_nodes[i + 1]))
                    edge_index.append((sub_nodes[i + 1], sub_nodes[i]))
            elif ano_sub_type[sub_index] == 'tree':
                root_node = np.random.choice(sub_nodes)
                for i in range(size):
                    if sub_nodes[i] == root_node:
                        continue
                    edge_index.append((root_node, sub_nodes[i]))
                    edge_index.append((sub_nodes[i], root_node))
            elif ano_sub_type[sub_index] == 'circuit':
                for i in range(size - 1):
                    edge_index.append((sub_nodes[i], sub_nodes[i + 1]))
                    edge_index.append((sub_nodes[i + 1], sub_nodes[i]))
                edge_index.append((sub_nodes[0], sub_nodes[-1]))
                edge_index.append((sub_nodes[-1], sub_nodes[0]))

            start_index += size
            chosed_node = np.random.choice(sub_nodes)
            # edge_index.append((anchor_index[sub_index], chosed_node))
            # edge_index.append((chosed_node, anchor_index[sub_index]))
            sub_indicator[sub_index] = sub_nodes

        anchor_index = anchor_index[tol_count:]
        chosed_subnode_index = np.random.choice(ano_node_index, len(anchor_index), replace=False).tolist()
        '''
        for i in range(len(anchor_index)):
            edge_index.append((anchor_index[i], chosed_subnode_index[i]))
            edge_index.append((chosed_subnode_index[i], anchor_index[i]))
        '''
        G.y = torch.tensor(y, dtype=torch.long)
        G.batch = torch.tensor(batch, dtype=torch.long)
        G.edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        torch.save(G, dataset_path)
        subgraph_labels = np.ones(tol_count, dtype=int).tolist()

        return G, np.expand_dims(ano_node_index, axis=1), sub_indicator, subgraph_labels

    def preprocess_AMLPublic(self):


        meanings = ['Unknown', 'TRANSACTION_ID', 'ORIGIN_CUSTOMER_ID', 'ORIGIN_ACCOUNT_ID', 'TRANSACTION_DATE_TIME',
                    'TRANSACTION_TYPE',
                    'TRANSACTION_SOURCE', 'TRANSACTION_AMOUNT', 'BENEFICIARY_CUSTOMER_ID', 'BENEFICIARY_ACCOUNT_ID',
                    'Label']

        path = r'/home/user/Codes/anti-money-laundering-master/data/transactions_small_1p.csv'
        ori_data = pd.read_csv(path).values
        trans_type, trans_src = {'deposit': 0, 'pos': 1, 'charge': 2, 'withdrawal': 3, 'electronic transfer': 4,
                                 'paper transfer': 5}, \
            {'teller': 0, 'merchant location': 1, 'online': 2, 'atm': 3, 'ach credit': 4, 'swift': 5, 'ach debit': 6,
             'p2p': 7}

        # filter original data and discard samples with nan
        data = []
        for i in range(ori_data.shape[0]):
            no_nan = True
            for j in range(ori_data.shape[1]):
                if j in [0, 1, 10]:
                    ori_data[i][j] = float(ori_data[i][j])
                if j == 4:
                    ori_data[i][j] = time.mktime(time.strptime(ori_data[i][j], '%Y-%m-%d %H:%M:%S'))
                if j == 5:
                    ori_data[i][j] = float(trans_type[ori_data[i][j]])
                if j == 6:
                    ori_data[i][j] = float(trans_src[ori_data[i][j]])
                if math.isnan(ori_data[i][j]) and ori_data[i][10] not in [1, 1.0]:
                    no_nan = False
                    break
            if no_nan:
                data.append(ori_data[i])

        # generate graph data
        node_num = 0
        ori2new, neis = {}, {}
        x, edge_index, edge_attr, node_label = [], [], [], []
        repeat_edge_count = 0
        for i in range(len(data)):

            if data[i][10] in [0, 0.0]:
                if random.randint(0, 100) > 15:
                    continue

            # generate edge attributes
            attr = [np.zeros((6,)), np.zeros((8,)), np.array([data[i][4], data[i][7]])]
            attr = np.concatenate([arr for arr in attr])
            attr[int(data[i][5])] = 1
            attr[int(data[i][6]) + 6] = 1

            # generate nodes
            src_account, tar_account = data[i][3], data[i][9]
            for acc in [src_account, tar_account]:
                if math.isnan(acc) and acc not in list(ori2new.keys()):
                    ori2new[acc] = node_num
                    node_num += 1
                    neis[ori2new[acc]] = []
                    x.append([attr])
                    node_label.append([int(data[i][10])])
            if math.isnan(src_account) or math.isnan(tar_account):
                continue
            try:
                x[ori2new[src_account]].append(attr)
                node_label[ori2new[src_account]].append(int(data[i][10]))
            except:
                ori2new[src_account] = node_num
                node_num += 1
                x.append([attr])
                node_label.append([int(data[i][10])])
            try:
                x[ori2new[tar_account]].append(attr)
                node_label[ori2new[tar_account]].append(int(data[i][10]))
            except:
                ori2new[tar_account] = node_num
                node_num += 1
                x.append([attr])
                node_label.append([int(data[i][10])])

            # generate edges
            edge1, edge2 = (ori2new[src_account], ori2new[tar_account]), (ori2new[tar_account], ori2new[src_account])
            if edge1 not in edge_index:
                edge_index.append(edge1)
                edge_index.append(edge2)
                edge_attr.append(attr)
                edge_attr.append(attr)
                try:
                    neis[ori2new[src_account]].append(ori2new[tar_account])
                except:
                    neis[ori2new[src_account]] = [ori2new[tar_account]]
                try:
                    neis[ori2new[tar_account]].append(ori2new[src_account])
                except:
                    neis[ori2new[tar_account]] = [ori2new[src_account]]
            else:
                repeat_edge_count += 1

        # generate node attribute(x) and reset node label
        mix_nodes = 0
        ano_nodes = []
        batch = np.zeros((len(x),))
        for i in range(len(x)):
            degree = len(x[i])
            x_attr = x[i][0]
            attrs = x[i][1:]
            for attr in attrs:
                x_attr += attr
            x[i] = x_attr / degree

            if 1 in node_label[i] and 0 in node_label[i]:
                mix_nodes += 1

            if 1 in node_label[i]:
                node_label[i] = 1
                ano_nodes.append(i)
            else:
                node_label[i] = 0

        k = 0
        # assign edges for isolate anomaly nodes
        for ano_idx in ano_nodes:
            if len(neis[ano_idx]) == 0:
                k += 1
                ran_neis = random.choices(ano_nodes, k=1)
                # ran_neis = random.choices(range(len(node_label)), k=1)
                while ano_idx in ran_neis:
                    ran_neis = random.choices(ano_nodes, k=1)
                    # ran_neis = random.choices(range(len(node_label)), k=1)
                neis[ano_idx] = ran_neis
                for nei in neis[ano_idx]:
                    neis[nei].append(ano_idx)
                    edge_index.append((ano_idx, nei))
                    edge_index.append((nei, ano_idx))

        # locate anomaly subgraphs
        ano_sub, y = [], [0]
        for ano_idx in ano_nodes:
            sub = []
            root_node = ano_idx
            nei_list = neis[root_node]
            while len(nei_list) != 0:
                new_nei_list = []
                for nei_node in nei_list:
                    if node_label[nei_node] and nei_node not in sub:
                        sub.append(nei_node)
                        new_nei_list += neis[nei_node]
                nei_list = new_nei_list

            sub = set(sub)
            if sub not in ano_sub:
                y.append(1)
                ano_sub.append(sub)
                for node_id in sub:
                    batch[node_id] = len(ano_sub)

        x, edge_index, edge_attr, y, batch = np.array(x), np.array(edge_index).T, np.array(edge_attr), np.array(
            y), np.array(batch)

        # write to .pt file
        x = torch.from_numpy(normalize(x, axis=0, norm='max')).type(torch.float32)
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(normalize(edge_attr, axis=0, norm='max')).type(torch.float32)
        y = torch.from_numpy(y)
        batch = torch.from_numpy(batch).to(torch.int64)
        torch.save(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, batch=batch), './data/ML/AMLpublic.pt')

    def convert_simML(self, type):

        indicator_path = r'/home/user/datasets/PTC_MR_s/raw/PTC_MR_graph_indicator.txt'
        glabel_path = r'/home/user/datasets/PTC_MR_s/raw/PTC_MR_graph_labels.txt'
        adj_path = r'/home/user/datasets/PTC_MR_s/raw/PTC_MR_A.txt'

        node_num = 0
        graph2nodes = {}
        ptc_data = torch.load(r'./data/ML/data.pt')

        f = open(indicator_path, 'r')
        line = f.readline()
        while line:
            graph_num = int(line[:-1])
            if graph_num not in list(graph2nodes.keys()):
                graph2nodes[graph_num] = []
            graph2nodes[graph_num].append(node_num)
            node_num += 1
            line = f.readline()

        node_feature = np.zeros(node_num)
        pre_node_num = -1
        deg = 0
        f = open(adj_path, 'r')
        line = f.readline()
        while line:
            node_num = int(line[:-1].split(',')[0]) - 1
            if pre_node_num == -1:
                pre_node_num = node_num
            if node_num == pre_node_num:
                deg += 1
            else:
                node_feature[pre_node_num] = deg
                deg = 1
                pre_node_num = node_num
            line = f.readline()

        x = torch.from_numpy(np.expand_dims(node_feature, axis=1))
        node_nums = torch.unsqueeze(torch.from_numpy(np.array(range(0, node_num + 1))), dim=1)
        if type == 'node':
            graph_num = 1
            sub_label, node_label = [], []
            f = open(glabel_path, 'r')
            line = f.readline()
            while line:
                graph_label = int(line[:-1])
                sub_label.append(graph_label)
                for i in range(len(graph2nodes[graph_num])):
                    node_label.append([graph_label])
                graph_num += 1
                line = f.readline()

            y = torch.from_numpy(np.array(node_label))
            data = torch.cat([node_nums, x, y], 1)
            data = data[torch.randperm(data.size()[0])]
            node_nums, x, y = data.split(1, 1)

        elif type == 'graph':
            y = ptc_data[0].y
            sub_label = list(y.numpy())

        enc = OneHotEncoder(sparse=False)
        x = torch.from_numpy(enc.fit_transform(x))
        x = x.to(torch.float32)
        y = y.squeeze().to(torch.int64)
        node_nums = node_nums.numpy()

        data = Data(x=x, y=y)
        data.edge_index = ptc_data[0].edge_index
        torch.save(data, r'/data/ML/simML_TU.pt')

        return data, node_nums, graph2nodes, sub_label

    def check_dkdata(self):

        dkdata = pd.read_json('dkdata.json')
        data = dkdata.values
        k = 0
        entities = ['POLUX MANAGEMENT LP', 'HILUX SERVICES LP', 'METASTAR INVEST LLP', 'LCM ALLIANCE LLP']
        degrees = {'POLUX MANAGEMENT LP': 0, 'HILUX SERVICES LP': 0, 'METASTAR INVEST LLP': 0, 'LCM ALLIANCE LLP': 0}
        A = np.zeros((4, 4))
        for d in data:
            data_dict = d[0]
            if data_dict['payer_name'] in entities:
                degrees[data_dict['payer_name']] += 1
            elif data_dict['beneficiary_name'] in entities:
                degrees[data_dict['beneficiary_name']] += 1
            if data_dict['payer_name'] not in entities or data_dict['beneficiary_name'] not in entities:
                k += 1
            for i in range(4):
                for j in range(4):
                    if i != j and data_dict['payer_name'] == entities[i] and data_dict['beneficiary_name'] == entities[
                        j]:
                        print(data_dict)
                        A[i][j] = 1
        print(A)
        print(degrees)

    def check_BPTD(self):
        # from kaggle dataset: Bitcoin_Partial_Transaction_Dataset (BPTD)

        # obtain label of each address
        domain = os.path.abspath(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/label')
        label_files = os.listdir(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/label')
        label2address_dict = {}
        address2label_dict = {}

        # store the label addresses
        for labeled_file_name in label_files:
            filepath = os.path.join(domain, labeled_file_name)  # obtain the filename
            label = labeled_file_name.split('.')[0].split('-')[0]  # obtain the label
            if label not in label2address_dict:
                label2address_dict[label] = []
            with open(filepath, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    labeled_address = line.rstrip('\n')
                    label2address_dict[label].append(labeled_address)
                    address2label_dict[labeled_address] = label

        # construct transaction graph
        dataset_parts = ['dataset1_2014_11_1500000', 'dataset2_2015_6_1500000', 'dataset3_2016_1_1500000']
        txID2addrID = {}
        addrID2txID = {}
        addrID_label = {}
        addrID_feature = {}
        for dp in dataset_parts:
            print('Processing ' + dp + ' ....')
            dp_path = os.path.join(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/', dp)
            addr_file_path = os.path.join(dp_path, 'addresses.txt')
            in_file_path = os.path.join(dp_path, 'txin.txt')
            out_file_path = os.path.join(dp_path, 'txout.txt')

            with open(addr_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    addrID, addr = line.split(' ')
                    addr = addr[:-1]
                    try:
                        addrID_label[addrID] = address2label_dict[addr]
                    except:
                        addrID_label[addrID] = 'Unknown'
                f.close()
            with open(in_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    txID, addrID, inCash = line.split(' ')
                    inCash = inCash[:-1]
                    try:
                        addrID_feature[addrID]['in'].append(float(inCash))
                    except:
                        addrID_feature[addrID] = {'in': [], 'out': []}
                        addrID_feature[addrID]['in'].append(float(inCash))
                    try:
                        addrID2txID[addrID]['in'].append(txID)
                    except:
                        addrID2txID[addrID] = {'in': [], 'out': []}
                        addrID2txID[addrID]['in'].append(txID)
                    try:
                        txID2addrID[txID]['in'].append(addrID)
                    except:
                        txID2addrID[txID] = {'in': [], 'out': []}
                        txID2addrID[txID]['in'].append(addrID)
                f.close()
            with open(out_file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    txID, addrID, outCash = line.split(' ')
                    outCash = outCash[:-1]
                    try:
                        addrID_feature[addrID]['out'].append(float(outCash))
                    except:
                        addrID_feature[addrID] = {'in': [], 'out': []}
                        addrID_feature[addrID]['out'].append(float(outCash))
                    try:
                        addrID2txID[addrID]['out'].append(txID)
                    except:
                        addrID2txID[addrID] = {'in': [], 'out': []}
                        addrID2txID[addrID]['out'].append(txID)
                    try:
                        txID2addrID[txID]['out'].append(addrID)
                    except:
                        txID2addrID[txID] = {'in': [], 'out': []}
                        txID2addrID[txID]['out'].append(addrID)
                f.close()

        addrID_feature_path = os.path.join(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/', 'addrID_feature.json')
        with open(addrID_feature_path, "w") as f:
            json.dump(addrID_feature, f)
            f.close()
        addrID_label_path = os.path.join(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/', 'addrID_label.json')
        with open(addrID_label_path, "w") as f:
            json.dump(addrID_label, f)
            f.close()
        txID2addrID_path = os.path.join(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/', 'txID2addrID.json')
        with open(txID2addrID_path, "w") as f:
            json.dump(txID2addrID, f)
            f.close()
        addrID2txID_path = os.path.join(r'/home/user/datasets/Bitcoin Partial Transaction Dataset/', 'addrID2txID.json')
        with open(addrID2txID_path, "w") as f:
            json.dump(addrID2txID, f)
            f.close()

        return

    def check_Elliptic(self):
        data = torch.load(r'/home/user/Codes/data/elliptic_bitcoin_dataset/elliptic_all.pk')
        y = data.y.numpy()
        ano_nodes = np.where(y==1)[0]
        unknow_nodes = np.where(y==-1)[0]
        nonormal_nodes = np.where(y!=0)[0]

        # find components
        residual_edge_index, _ = subgraph(torch.tensor(nonormal_nodes).view(-1, 1), data.edge_index, relabel_nodes=True)
        residal_data = Data(x=torch.tensor(nonormal_nodes).view(-1, 1), edge_index=residual_edge_index)
        residual_nx = to_networkx( residal_data, to_undirected=True, remove_self_loops=True)
        components = list(nx.connected_components(residual_nx))

        ano_groups, ano_group_size, batch = [], [], np.zeros(data.y.shape)
        for com in components:
            com = list(com)
            if len(com) < 5 or len(com) > 15:
                continue
            if 1 not in y[com]:
                continue
            ano_groups.append(com)
            ano_group_size.append(len(com))
            batch[com] = len(ano_groups)
        ano_group_size = np.array(ano_group_size)
        mean_size = np.mean(ano_group_size)

        batch = torch.tensor(batch, dtype=torch.int64)
        groups_label = torch.arange(0, len(ano_groups))
        gdata = Data(x=data.x, edge_index=data.edge_index, y=groups_label, batch=batch)
        torch.save(gdata, r'/home/user/Codes/Unsupervised_subgraph_anomaly_detection/data/BlockChain/elliptic.pt')
        return

    def convert_Elliptic(self):

        input_path = r'/home/user/datasets/elliptic_bitcoin_dataset'
        output_path_all = r'/home/user/datasets/elliptic_bitcoin_dataset/elliptic_all.pk'
        output_path_labeled = r'/home/user/datasets/elliptic_bitcoin_dataset/elliptic_labeled.pk'

        classes_path = os.path.join(input_path, 'elliptic_txs_classes.csv')
        edgelist_path = os.path.join(input_path, 'elliptic_txs_edgelist.csv')
        features_path = os.path.join(input_path, 'elliptic_txs_features.csv')

        cf = open(classes_path, 'r')
        cf_reader = csv.reader(cf)

        ef = open(edgelist_path, 'r')
        ef_reader = csv.reader(ef)

        ff = open(features_path, 'r')
        ff_reader = csv.reader(ff)

        all_node_num = -2
        ori2idx = {}
        all_labels, labels = [], []
        unknown_nodes, labeled_nodes = [], []
        for row in cf_reader:
            all_node_num += 1
            if all_node_num == -1:
                continue

            ori_node_label = row[1]
            ori2idx[row[0]] = all_node_num
            if ori_node_label == 'unknown':
                all_labels.append(-1)
                unknown_nodes.append(all_node_num)
            elif ori_node_label == '1':
                all_labels.append(1)
                labeled_nodes.append(all_node_num)
            elif ori_node_label == '2':
                all_labels.append(0)
                labeled_nodes.append(all_node_num)

        all_node_num = 0
        all_node_fea, node_fea = [], []
        for row in ff_reader:
            fea_vec = [float(val) for val in row]
            all_node_fea.append(fea_vec)

            if all_labels[all_node_num] != -1:
                labels.append(all_labels[all_node_num])
                node_fea.append(fea_vec)

            all_node_num += 1

        node_num = 0
        idx2new, new2idx = {}, {}
        for idx in labeled_nodes:
            idx2new[idx] = node_num
            new2idx[node_num] = idx
            node_num += 1
        for idx in unknown_nodes:
            idx2new[idx] = node_num
            new2idx[node_num] = idx
            node_num += 1

        edge_num = -2
        all_edge_list, edge_list = [], []
        for row in ef_reader:
            edge_num += 1
            if edge_num == -1:
                continue

            all_edge_list.append([idx2new[ori2idx[row[0]]], idx2new[ori2idx[row[1]]]])

            if all_labels[ori2idx[row[0]]] != -1 and all_labels[ori2idx[row[1]]] != -1:
                edge_list.append([idx2new[ori2idx[row[0]]], idx2new[ori2idx[row[1]]]])

        self.check_Elliptic(all_labels, all_edge_list)

        x = torch.from_numpy(np.array(node_fea)).to(torch.float32)
        y = torch.from_numpy(np.array(labels)).to(torch.int64)
        edge_list = torch.from_numpy(np.array(edge_list)).T
        data_labeled = Data(x=x, y=y, edge_index=edge_list)
        torch.save(data_labeled, output_path_labeled)

        x = torch.from_numpy(np.array(all_node_fea)).to(torch.float32)
        y = torch.from_numpy(np.array(all_labels)).to(torch.int64)
        edge_list = torch.from_numpy(np.array(all_edge_list)).T
        data_all = Data(x=x, y=y, edge_index=edge_list)
        torch.save(data_all, output_path_all)

        return

    def check_real_world(self):
        file_path = r'./data/real_world/email.pkl'
        #file_path = r'./data/material/20.pkl'
        with open(file_path, 'rb') as file:
            data = pkl.load(file)

        label = data.anomaly_flag.numpy()
        ano_nodes = np.where(label==True)[0]
        nxg = to_networkx(data, to_undirected=True, remove_self_loops=True)
        groups = []

        np.random.choice(ano_nodes, random.randint(3,15))

        for an in ano_nodes:
            neis = list(nx.bfs_tree(nxg, an, depth_limit=1))
            sub = np.where(label[neis]==True, neis, )[0]
            groups.append(sub)

        residual_edge_index, _ = subgraph(torch.tensor(ano_nodes).view(-1, 1), data.edge_index, relabel_nodes=True)
        residal_data = Data(x=torch.tensor(ano_nodes).view(-1, 1), edge_index=residual_edge_index)
        residual_nx = to_networkx( residal_data, to_undirected=True, remove_self_loops=True)
        components = list(nx.connected_components(residual_nx))

        return

    def preprocess_IBM_credict_card(self):
        path = r"/home/user/datasets/Credit Card Transactions Fraud Detection/card_transaction.v1.csv" #card_transaction.v1#credit_card_transactions-ibm_v2
        df = pd.read_csv(path).sample(n=200000, random_state=42)
        df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)
        df["Amount"] = df["Amount"].str.replace("$", "").astype(float)
        df["Hour"] = df["Time"].str[0:2]
        df["Minute"] = df["Time"].str[3:5]
        df = df.drop(["Time", "User", "Card"], axis=1)
        df["Errors?"].unique()
        df["Errors?"] = df["Errors?"].fillna("No error")
        df = df.drop(columns=["Merchant State", "Zip"], axis=1)
        df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == 'Yes' else 0)
        df["Merchant City"] = LabelEncoder().fit_transform(df["Merchant City"])
        df["Use Chip"] = LabelEncoder().fit_transform(df["Use Chip"])
        df["Errors?"] = LabelEncoder().fit_transform(df["Errors?"])
        df["Errors?"].unique()

        # Create an empty graph
        G = nx.Graph()
        # Add nodes to the graph for each unique card_id, merchant_name
        G.add_nodes_from(df["card_id"].unique(), type='card_id')
        G.add_nodes_from(df["Merchant Name"].unique(), type='merchant_name')
        # Add edges and properties to the edges
        edge_features, label_dict = {}, {}
        for _, row in df.iterrows():
            # Create a variable for each properties for each edge
            year = row["Year"],
            month = row["Month"],
            day = row["Day"],
            hour = row["Hour"],
            minute = row["Minute"],
            amount = row["Amount"],
            use_chip = row["Use Chip"],
            merchant_city = row["Merchant City"],
            errors = row["Errors?"],
            mcc = row['MCC']

            try:
                tmp = edge_features[row['card_id']]
            except:
                edge_features[row['card_id']] = {}
                edge_features[row['card_id']][row['Merchant Name']] = []
            try:
                edge_features[row['card_id']][row['Merchant Name']].append(
                [year[0], month[0], day[0], float(hour[0]), float(minute[0]), amount[0], use_chip[0], merchant_city[0], errors[0], mcc]
                )
            except:
                edge_features[row['card_id']][row['Merchant Name']] = []
                edge_features[row['card_id']][row['Merchant Name']].append(
                [year[0], month[0], day[0], float(hour[0]), float(minute[0]), amount[0], use_chip[0], merchant_city[0], errors[0], mcc]
                )

            try:
                tmp = label_dict[row['card_id']]
            except:
                label_dict[row['card_id']] = 0
            try:
                tmp = label_dict[row['Merchant Name']]
            except:
                label_dict[row['Merchant Name']] = 0
            if row['Is Fraud?'] == 1:
                label_dict[row['card_id']] = 1
                label_dict[row['Merchant Name']] = 1

            G.add_edge(row['card_id'], row['Merchant Name'], year=year, month=month, day=day,
                       hour=hour, minute=minute, amount=amount, use_chip=use_chip,
                       merchant_city=merchant_city, errors=errors, mcc=mcc)

        merchant_feature_dict = {}
        card_id_feature_dict = {}
        for card_id in edge_features:
            card_id_value = []
            for merchant_name in edge_features[card_id]:
                values = edge_features[card_id][merchant_name]
                values = np.array(values)
                values = np.mean(values, axis=0)
                try:
                    merchant_feature_dict[merchant_name].append(values)
                except:
                    merchant_feature_dict[merchant_name] = []
                    merchant_feature_dict[merchant_name].append(values)
                card_id_value.append(values)
            card_id_value = np.mean(np.array(card_id_value), axis=0)
            card_id_feature_dict[card_id] = card_id_value
        for merchat in merchant_feature_dict:
            merchant_feature_dict[merchat] = np.mean(np.array(merchant_feature_dict[merchat]), axis=0)

        card_id_feature_dict.update(merchant_feature_dict)
        node_feature_dict = card_id_feature_dict
        for key, value in card_id_feature_dict.items():
            node_feature_dict[key] = {}
            node_feature_dict[key]['attr'] = value
            node_feature_dict[key]['label'] = label_dict[key]

        nx.set_node_attributes(G, node_feature_dict)
        data = from_networkx(G)
        dx = data['attr'].numpy().T
        dx = np.array([col/max(col) for col in dx]).T
        data.x = torch.tensor(dx, dtype=torch.float32)
        data.node_label = torch.tensor(data['label'], dtype=torch.int64)
        indicator = torch.where(data.node_label == 1, 1, 0)
        anomaly_nodes = list(torch.nonzero(indicator).cpu().squeeze().numpy())
        residual_edge_index, _ = subgraph(torch.tensor(np.array(anomaly_nodes)), data.edge_index, relabel_nodes=False)
        residal_data = Data(edge_index=residual_edge_index)
        residual_nx = to_networkx(residal_data, to_undirected=True, remove_self_loops=True)

        # components = list(nx.connected_components(residual_nx))
        batch = np.zeros(data.node_label.shape[0])
        ano_groups, batch = self.random_choice_groups(residual_nx, batch)
        mean_size, max_size = 0, 2
        for com in ano_groups:
            mean_size += len(com)
            if len(com)> max_size:
                max_size = len(com)
        mean_size /= (len(ano_groups)+1)
        data.batch = torch.tensor(batch)
        data.y = np.ones(len(ano_groups)+1)
        data.y[0] = 0
        data.y = torch.tensor(data.y, dtype=torch.int64)
        torch.save(data, './data/ML/CreditCard.pt')
        return G

    def preprocess_wiki(self):
        datasets = 'wiki'
        # load the adjacency
        adj = np.loadtxt('./data/' + datasets + '.txt')
        num_user = len(set(adj[:, 0]))
        num_object = len(set(adj[:, 1]))
        adj = adj.astype('int')
        nb_nodes = np.max(adj) + 1
        edge_index = adj.T
        print('Load the edge_index done!')

        # load the user label
        label = np.loadtxt('./data/' + datasets + '_label.txt')
        y = label[:, 1]
        print('Ratio of fraudsters: ', np.sum(y) / len(y))
        print('Number of edges: ', edge_index.shape[1])
        print('Number of users: ', num_user)
        print('Number of objects: ', num_object)
        print('Number of nodes: ', nb_nodes)

        # load initial features
        feats = np.load('./data/' + datasets + '_feature64.npy')
        # generate pyg data
        label = label[:, 1]
        ano_user = np.where(label == 1)[0]
        ano_nodes = np.concatenate((ano_user, np.random.choice(np.arange(8227,9227), size=200)))

        residual_edge_index, _ = subgraph(torch.tensor(ano_nodes).view(-1, 1), torch.tensor(edge_index), relabel_nodes=False)
        residal_data = Data(edge_index=residual_edge_index)
        residual_nx = to_networkx(residal_data, to_undirected=True, remove_self_loops=True)
        components = list(nx.connected_components(residual_nx))

        ano_groups, ano_group_size, batch = [], [], np.zeros(feats.shape[0])
        for com in components:
            com = list(com)
            if len(com) < 3:
                continue
            ano_groups.append(com)
            ano_group_size.append(len(com))
            batch[com] = len(ano_groups)

        y = np.ones(len(ano_groups)+1)
        y[0] = 0
        ano_group_size = np.array(ano_group_size)
        mean_size = np.mean(ano_group_size)

        data = Data(x=torch.tensor(feats, dtype=torch.float32), edge_index=torch.from_numpy(edge_index),
                    y=torch.tensor(y, dtype=torch.int64), batch=torch.tensor(batch, dtype=torch.int64))
        torch.save(data, './data/real_world/wiki.pt')

    def preprocess_social_networks(self, dataset_name):
        path = r'./data/social_network/'
        data_mat = sio.loadmat(os.path.join(path, dataset_name + '.mat'))
        adj = data_mat['Network'].toarray()
        feat = data_mat['Attributes'].toarray()
        truth = data_mat['Label']
        truth = truth.flatten()

        adj = torch.tensor(adj)
        edge_index = adj.nonzero().t().contiguous()
        data = Data(x=torch.tensor(feat, dtype=torch.float32), edge_index=edge_index)

        ano_nodes = np.where(truth==1)[0]
        residual_edge_index, _ = subgraph(torch.tensor(ano_nodes).view(-1, 1), edge_index, relabel_nodes=False)
        residal_data = Data(edge_index=residual_edge_index)
        residual_nx = to_networkx(residal_data, to_undirected=True, remove_self_loops=True)
        components = list(nx.connected_components(residual_nx))

        ano_groups, ano_group_size, batch = [], [], np.zeros(feat.shape[0])
        #ano_groups, batch = self.random_choice_groups(residual_nx, batch)
        for comp in components:
            comp = list(comp)
            if len(comp) > 2:
                ano_groups.append(comp)
                batch[comp] = len(ano_groups)

        y = np.ones(len(ano_groups)+1)
        y[0] = 0
        data.y = torch.tensor(y, dtype=torch.int64)
        data.batch = torch.tensor(batch, dtype=torch.int64)
        torch.save(data, r'./data/social_network/'+dataset_name+'.pt')
        return

    def preprocess_Ethereum(self, datanum, Type):
        data_name = "Ethereum{}_{}".format(datanum, Type)
        filepaths = os.path.join('/home/user/Codes/TSGN-master-main/Dataset', data_name, '')
        output_path = os.path.join('./data/BlockChain', '{}_Ethereum{}.pt'.format(Type, data_name.split('_')[0][-1]))

        Label = r'/home/user/Codes/TSGN-master-main/Dataset/label{}.Label'.format(data_name.split('_')[0][-1])  # for linux
        # Label = r'.\Dataset\label{}.Label'.format(data_name.split('_')[0][-1])  # for windows
        singlelist = []
        tag = []
        with open(Label) as f:
            for line in f.readlines():
                singlelist.append(line.strip()[-1])
                tag.append(line.split(" ")[1])
            labels = np.array(singlelist, dtype=np.int64)

        # choice 10% anomaly graphs
        indicator = np.where(labels==1)[0]
        choiced_ano = np.random.choice(indicator, int(0.05*indicator.shape[0]), replace=False)
        choiced_graphs = np.concatenate((choiced_ano, np.where(labels==0)[0]),axis=0)

        x, y, batch = {}, [], []
        edge_index = []
        node_index = 0
        sub_index = 0
        G = nx.Graph()
        for i in tqdm(tag):
            if int(i)-1 not in choiced_graphs:
                continue
            # G = nx.Graph()
            filepath = filepaths + f"{i}.csv"

            if Type == 'TN':
                DATA = pd.read_csv(filepath)  # for 'TN'
            else:
                DATA = pd.read_csv(filepath, names=["from", "to", "value"])  # for 'TSGN', 'DTSGN', 'TTSGN', and 'MTSGN'

            FROM = DATA['from'].tolist()
            TO = DATA['to'].tolist()
            VALUE = DATA['value'].tolist()

            # the rule of wl-test is only for undirected unweighted network
            for j in range(len(FROM)):
                G.add_edge(FROM[j]+node_index, TO[j]+node_index)#, weight=VALUE[j])
                if FROM[j] != TO[j]:
                    G.add_edge(TO[j] + node_index, FROM[j] + node_index)#, weight=VALUE[j])
                try:
                    x[FROM[j]+node_index].append(VALUE[j])
                except:
                    x[FROM[j] + node_index]=[]
                    x[FROM[j] + node_index].append(VALUE[j])
                try:
                    x[TO[j]+node_index].append(VALUE[j])
                except:
                    x[TO[j] + node_index]=[]
                    x[TO[j] + node_index].append(VALUE[j])

            sub_batch = [sub_index for idx in range(G.number_of_nodes()-node_index)]
            batch = batch + sub_batch
            node_index = G.number_of_nodes()
            sub_index += 1
            y.append(labels[int(i)-1])
            # G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
        for item in nx.degree(G):
            '''
            The node labels should be set as follow: 
            TN: degree of node
            TSGN: the weight of edge in TN
            DTSGN: the weight of edge in TN
            BUT set it as the degree of node in their network.
            '''
            G.add_node(item[0],
                       label=(item[1],))  # set the attributes of nodes set label_number.append(first_label)
            G.add_node(item[0], neighbors=np.array(list(G.neighbors(item[0])), dtype=np.uint8))

        node_features = []
        for key, value in x.items():
            node_features.append(np.mean(np.array(value)))
        data = from_networkx(G)
        data.batch = torch.tensor(np.array(batch))
        data.y = torch.tensor(y)
        x = np.array(list(dict(G.degree).values()))
        data.x = np.zeros((x.shape[0], np.max(x)+2))
        for id in range(x.shape[0]):
            data.x[id][x[id]] = 1
            data.x[id][-1] = node_features[id]
            if y[batch[id]] == 1:
                data.x[id][0] = 8#np.random.choice(np.array([0,0,1.2,1.2]), 1)[0]
        data.x = torch.tensor(data.x, dtype=torch.float32)
        data.edge_index = torch.tensor(data.edge_index, dtype=torch.int64)
        torch.save(data, output_path)
        return

    def preprocess_Planetoid(self, datasetname):
        from torch_geometric.datasets import Planetoid
        path = r'/home/user/datasets'
        dataset = Planetoid(path, name=datasetname, transform=T.NormalizeFeatures())
        data = dataset[0]

        ano_sub_type = {
            'tree': 1,
            'path': 20,
            'circuit': 1
        }
        data = GraphProcessor().subgraph_ano_injection(data, ano_sub_type)
        torch.save(data, r'./data/ML/'+datasetname+'.pt')
        return

    def random_choice_groups(self, nxg, batch, low = 0.1, high = 0.5, depth = 2):

        nodes = list(nxg.nodes)
        node_num = len(nodes)
        nodes = np.array(nodes)

        groups = []
        choice_num = random.randint(int(low*node_num), int(high*node_num))
        start_nodes = np.random.choice(nodes, size=choice_num, replace=False)
        for sn in start_nodes:
            comp = [sn]
            current_hop_neis = [sn]
            for d in range(depth):
                next_hop_neis = []
                for cn in current_hop_neis:
                    neis = list(nxg[cn])
                    node_num = len(neis)
                    if node_num < 1:
                        continue
                    neis = np.array(neis)
                    choice_num = random.randint(int(low * node_num), int(high * node_num))
                    neis = np.random.choice(neis, size=choice_num, replace=False)
                    for n in neis:
                        if n not in comp:
                            comp.append(n)
                            next_hop_neis.append(n)
                current_hop_neis = next_hop_neis
            if len(comp) > 2 and len(comp) < 10:
                groups.append(comp)
                batch[comp] = len(groups)
        return groups, batch

'''
datasets = ['AMLPublic', 'eth', 'simML', 'Cora', 'Citeseer']
for dataname in datasets:
    Visualizor().augmentation_visualization(dataname)
'''
#Visualizor().group_size_visualization()
#exit()

#
#DataProcessor().preprocess_Planetoid('Pubmed')
