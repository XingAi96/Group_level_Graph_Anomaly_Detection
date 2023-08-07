import torch
from utils import *
from tqdm import tqdm
from initializer import *
from GCL.eval import get_split, SVMEvaluator
from torch_geometric.utils import to_dense_adj
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def filter_subgraph(args, data, anomaly_score, device, sample_type='connected_components'):

    # obtain the candidate subgraph
    threshold = np.percentile(anomaly_score, q=args.q)

    if sample_type == 'connected_components':
        residual_edge_index, _ = subgraph(
            torch.arange(data.num_nodes)[anomaly_score > threshold].to(device),
            data.edge_index,
            relabel_nodes=True
        )
        residal_data = Data(
            # x=torch.tensor(mean_error_list[mean_error_list > threshold]).view(-1, 1),
            x=data.x[anomaly_score > threshold],
            edge_index=residual_edge_index,
        )
        # find components
        residual_nx = to_networkx(
            residal_data,
            to_undirected=True,
            remove_self_loops=True
        )
        batch = torch.zeros(residal_data.num_nodes, dtype=torch.long)
        components = list(nx.connected_components(residual_nx))
        y, batch, is_mix, complete_ratio, comp_size = GraphProcessor().relabel(data, components, batch)
        residal_data.batch = batch
        residal_data.y = torch.from_numpy(np.array(y))
        residal_data.to(data.x.device)

        return residal_data, is_mix, complete_ratio

    elif sample_type == 'center_node':
        candi_groups, residal_data, sub_size = GraphProcessor().sample_sub(data, anomaly_score, threshold)
        return residal_data, candi_groups, sub_size


def train_GAE(args, data, model, optimizer=None, is_training=True):
    x, edge_index, A = data.x, data.edge_index, data.A

    if is_training:
        model.train()
        with tqdm(total=args.gcl_epochs, desc='(GAE)') as pbar:
            for epoch in range(1, args.gae_epochs):
                stru_recon, attr_recon = model(x, edge_index)
                stru_score = torch.square(stru_recon ** 3 - A).sum(1)
                # stru_score = (torch.square(stru_recon - A).sum(1) + torch.square(stru_reconp - Ap).sum(1))/2
                attr_score = torch.square(attr_recon - x).sum(1)
                score = args.alpha * stru_score + (1 - args.alpha) * attr_score
                loss = score.mean()
                total_error = score.clone().detach().cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

    else:
        with torch.no_grad():
            A = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
            stru_recon, attr_recon = model(data.x, data.edge_index)
            stru_score = torch.square(stru_recon ** 3 - A).sum(1).sqrt()
            attr_score = torch.square(attr_recon - data.x).sum(1).sqrt()
            score = args.alpha * stru_score + (1 - args.alpha) * attr_score
            total_error = score.detach().cpu().numpy()

    return total_error


def train_GCL(args, encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list =\
    dataloader[0], dataloader[1], dataloader[2], dataloader[3], dataloader[4]

    for idx in range(len(batch_list)):
        data = batch_list[idx]
        data = data.to('cuda')
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        _, g0, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch, cycle_edges_list[idx],
                                           tree_root_list[idx], path_middle_list[idx], one_degree_list[idx])
        g0, g1, g2 = [encoder_model.encoder.project(g) for g in [g0, g1, g2]]
        # loss, _ = contrast_model(g1=g1, g2=g2, batch=data.batch)

        # approximate mutual information via a model 
        inner_epochs = args.inner_epochs
        optimizer_local = torch.optim.Adam(contrast_model.parameters(), lr=args.inner_lr)
        for j in range(0, inner_epochs):
            optimizer_local.zero_grad()

            shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[torch.randperm(g2.shape[0])]
            joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
            margin1, margin2 = contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
            mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
                 (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))

            local_loss = mi
            local_loss.backward(retain_graph=True)
            optimizer_local.step()

        shuffle_g0, shuffle_g1, shuffle_g2 = g0[torch.randperm(g0.shape[0])], g1[torch.randperm(g1.shape[0])], g2[
            torch.randperm(g2.shape[0])]
        joint1, joint2 = contrast_model(g1, g2), contrast_model(g0, g1)
        margin1, margin2 = contrast_model(g1, shuffle_g2), contrast_model(g0, shuffle_g1)
        mi = - (torch.mean(joint1) - torch.log(torch.mean(torch.exp(margin1)))) + \
             (torch.mean(joint2) - torch.log(torch.mean(torch.exp(margin2))))
        loss = mi

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss, [encoder_model, contrast_model]


def train(args, data, device):

    # inizialize models
    GAE, GCL, opt_gae, opt_gcl = initialize_model(args)

    # train GAE to locate subgraph
    errors = train_GAE(args, data, GAE, optimizer=opt_gae, is_training=True)

    # filter
    residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors, device, args.sample_type)

    # preprocessing for locate critical edges and nodes of each batch
    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
    for rdata in [residal_data]:
        cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().sample_aug_sub(rdata)
        batch_list.append(rdata)
        cycle_edges_list.append(cycle_edges)
        tree_root_list.append(tree_root_nodes)
        path_middle_list.append(del_edge_index)
        one_degree_list.append(one_degree_nodes)

    # train GCL
    with tqdm(total=args.gcl_epochs, desc='(GCL)') as pbar:
        for epoch in range(1, args.gcl_epochs):
            loss, GCL = train_GCL(args, GCL[0], GCL[1],
        [batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list], opt_gcl)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    return GAE, GCL, batch_list


def test(args, GAE, GCL, data, device):
    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import f1_score
    from sklearn.svm import OneClassSVM
    from pyod.models.abod import ABOD
    from pyod.models.cblof import CBLOF
    from pyod.models.ecod import ECOD
    from PyNomaly import loop
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import roc_auc_score

    GCL_encoder, contrast_model = GCL[0], GCL[1]
    GAE.eval()
    GCL_encoder.eval()

    errors = train_GAE(args, data, GAE, optimizer=None, is_training=False)
    residal_data, candi_groups, sub_size = filter_subgraph(args, data, errors, device, args.sample_type)

    batch_list, cycle_edges_list, tree_root_list, path_middle_list, one_degree_list = [], [], [], [], []
    for rdata in [residal_data]:
        cycle_edges, tree_root_nodes, del_edge_index, one_degree_nodes = GraphProcessor().sample_aug_sub(rdata)
        batch_list.append(rdata)
        cycle_edges_list.append(cycle_edges)
        tree_root_list.append(tree_root_nodes)
        path_middle_list.append(del_edge_index)
        one_degree_list.append(one_degree_nodes)
    # _, g, _, _, g1, g2 = GCL_encoder(residal_data.x, residal_data.edge_index, residal_data.batch,
    #            cycle_edges_list[0], tree_root_list[0], path_middle_list[0], one_degree_list[0])

    # skip TPGCL
    groups = []
    for gid in range(residal_data.y.shape[0]):
        indicator = torch.where(residal_data.batch == gid, 1, 0)
        if np.sum(indicator.cpu().numpy()) == 0 or len(list(indicator.cpu().numpy())) == 0:
            embedding = np.zeros(residal_data.x.cpu().numpy().shape[1])
        else:
            try:
                node_list = list(torch.nonzero(indicator).cpu().squeeze().numpy())
                embedding = residal_data.x[node_list].cpu().numpy()
                embedding = np.mean(embedding, axis=0)
            except:
                embedding = np.zeros(residal_data.x.cpu().numpy().shape[1])
        groups.append(embedding)
    groups = np.array(groups)
    g = torch.from_numpy(groups)

    # _, g, _, _, _, _ = GCL_encoder(residal_data.x, residal_data.edge_index, residal_data.batch)

    x, y = [], []
    x.append(g)
    y.append(residal_data.y)
    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0).detach().cpu().numpy()

    # Visualizor().embedding_visualization(x,y,args.real_world_name)

    '''
    print(sum(is_mix), len(is_mix)-sum(is_mix))
    print('<=0.1 : ' + str(com_ratio[0]))
    print('0.1-0.4 : ' + str(com_ratio[1]))
    print('0.4-0.7 : ' + str(com_ratio[2]))
    print('0.7-0.999 : ' + str(com_ratio[3]))
    print('1.0 : ' + str(com_ratio[4]))

    print('<=0.1 : ' + str(red_ration[0]))
    print('0.1-0.4 : ' + str(red_ration[1]))
    print('0.4-0.7 : ' + str(red_ration[2]))
    print('0.7-0.999 : ' + str(red_ration[3]))
    print('1.0 : ' + str(red_ration[4]))
    '''
    split = get_split(num_samples=x.shape[0], train_ratio=0.8, test_ratio=0.1)

    keys = ['train', 'test', 'valid']
    objs = [x, y]
    x_train, x_test, x_val, y_train, y_test, y_val = [obj[split[key]] for obj in objs for key in keys]

    #cls = IsolationForest(random_state=0).fit(x_train)
    #cls = EllipticEnvelope(contamination=0.1).fit(x_train)
    #cls = LocalOutlierFactor().fit(x_train)
    #cls = OneClassSVM().fit(x_train)
    #m = loop.LocalOutlierProbability(x, use_numba=True, progress_bar=True).fit()
    #scores = m.local_outlier_probabilities
    #cls = DBSCAN().fit(x_train)
    #cls = ABOD(contamination=1e-1,n_neighbors=10).fit(x_train)
    #cls = CBLOF(n_clusters=2, contamination=0.3).fit(x_train)
    #x_train, y_train = x, y
    cls = ECOD(contamination=args.contamination).fit(x)

    y_score, y_pre = cls.decision_scores_, cls.labels_
    test_macro = f1_score(y, y_pre, average='macro')
    test_micro = f1_score(y, y_pre, average='micro')
    try:
        auc = roc_auc_score(y, y_score)
    except:
        auc = -1
    cr = CR_calculator(data, candi_groups, y_score, y_pre)

    result = {'micro_f1': test_micro, 'macro_f1': test_macro, 'auc': auc, 'cr': cr, 'comp_size': sub_size}
    print(result)
    '''
    result = SVMEvaluator(linear=True)(x, y, split)
    result['auc'] = 0
    result['cr'] = cr
    result['comp_size'] = comp_size
    print(result)
    '''
    return result