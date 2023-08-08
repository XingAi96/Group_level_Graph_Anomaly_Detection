import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import *


class GraphProcessor:

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

    def pattern_search(self, pyg):
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

    def anomaly_group_injector(self, G: Data, ano_sub: dict):
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


def CR_calculator(data, candi_groups, y_pre):
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
    cr = 0.5*(cr1+cr2)
    return cr
