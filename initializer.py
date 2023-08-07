
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GINConv, global_add_pool
from GAE.encoder import *
from GAE.decoder import *
from GAE.model import *
from GAE.smgnn import *
from GAE.dominant import *
import GCL.losses as L
import GCL.augmentors as A
from GCL.augmentors.augmentor import Graph
from GCL.models import DualBranchContrast


def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=32):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, int(project_dim/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(project_dim / 2), int(project_dim / 4))
        )

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index.type(torch.int64))
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch, cycle_edges_list=None,
                tree_root_list=None, path_middle_list=None, one_degree_list=None):

        aug1, aug2 = self.augmentor
        if cycle_edges_list != None or tree_root_list != None or \
                tree_root_list != None or path_middle_list != None:
            if isinstance(aug1, A.SubIncreasing) or isinstance(aug1, A.SubDecreasing):
                x1, edge_index1, edge_weight1, batch1 = \
                    aug1.augment(Graph(x, edge_index, None), batch, cycle_edges_list,
                                 tree_root_list, path_middle_list, one_degree_list)
                z1, g1 = self.encoder(x1, edge_index1, batch1)
            else:
                x1, edge_index1, edge_weight1 = aug1(x, edge_index)
                z1, g1 = self.encoder(x1, edge_index1, batch)

            if isinstance(aug2, A.SubIncreasing) or isinstance(aug2, A.SubDecreasing):
                x2, edge_index2, edge_weight2, batch2 = \
                    aug2.augment(Graph(x, edge_index, None), batch, cycle_edges_list,
                                 tree_root_list, path_middle_list, one_degree_list)
                z2, g2 = self.encoder(x2, edge_index2, batch2)
            else:
                x2, edge_index2, edge_weight2 = aug2(x, edge_index)
                z2, g2 = self.encoder(x2, edge_index2, batch)

        else:
            z1, z2, g1, g2 = None, None, None, None

        z, g = self.encoder(x, edge_index, batch)

        return z, g, z1, z2, g1, g2


class ContrastiveModel(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(ContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, y):
        x, y = F.normalize(x, p=2, dim=0), F.normalize(y, p=2, dim=0)
        h1 = F.relu(self.fc1(x)+self.fc2(y))
        h2 = self.fc3(h1)
        return h2

def initialize_model(args):

    # GAE parameters
    gae_out_channels, gae_num_features, gae_num_layer = \
        args.gae_out_channels, args.gae_num_features, args.gae_num_layer
    gae_embedding_channels, gae_hidden_channels = \
    args.gae_embedding_channels, args.gae_hidden_channels

    # GAE model
    GAEmodel = DOMINANT_MODEL(gae_num_features, gae_hidden_channels, gae_num_layer, nn.ReLU)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GAEmodel = GAEmodel.to(device)

    # GCL parameters
    aug1, aug2 = args.aug1, args.aug2
    gcl_input_dim, gcl_hidden_dim, gcl_num_layer, gcl_output_dim = \
        args.gcl_input_dim, args.gcl_hidden_dim, args.gcl_num_layer, args.gcl_output_dim

    # GCL
    gconv = GConv(input_dim=gcl_input_dim, hidden_dim=gcl_hidden_dim, num_layers=gcl_num_layer,
                  output_dim=gcl_output_dim).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    #contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
    contrast_model = ContrastiveModel(gcl_output_dim, gcl_output_dim, hidden_dim=32).to(device)
    GCLmodel = [encoder_model, contrast_model]

    # inizialize optimizers
    opt_gae = torch.optim.Adam(GAEmodel.parameters(), lr=args.gae_lr)
    opt_gcl = Adam(encoder_model.parameters(), lr=args.gcl_lr)

    return GAEmodel, GCLmodel, opt_gae, opt_gcl