import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, Sequential
from pyod.models.base import BaseDetector
from sklearn.metrics import roc_auc_score
from typing import Union, List, Tuple

class DOMINANT_MODEL(nn.Module):
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_layers: int,
            act,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.act = act

        self.enc = GCN(n_layers, n_input, n_hidden, n_hidden, act)
        self.attr_dec = GCN(1, n_hidden, ..., n_input, act)

    def forward(self, x, edge_index):
        z = self.enc(x, edge_index)
        return z @ z.T, self.attr_dec(z, edge_index)


class DOMINANT(BaseDetector):
    '''
    Interface of "Deep Anomaly Detection on Attributed Networks"(DOMINANT) model.

    Parameters
    ----------
    n_hidden : Union[List[int], Tuple[int], int], default=64
        Size of hidden layers. `n_hidden` can be list or tuple of `int`, or just `int`, which means all hidden layers has same size.
    n_layers : int, default=3
        Number of GCN encoder layers.
    act : default=nn.ReLU
        Activation function of each layer. Class name should be pass just like the default parameter `nn.ReLU`.
    alpha : float, default=0.5
        The weight of structural anomaly score, 1-alpha is the weight of attributed anomaly score correspondingly.
    lr : float, default=0.005
        The learning rate of optimizer (Adam).
    weight_decay : float, default=0.
        The weight decay parameter of optimizer (Adam).
    epoch : int, default=5
        Training epoches of DOMINANT.
    verbose : bool, default=False
        Whether to print training log, including training epoch and training loss (and ROC_AUC if pass label when fitting model).
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the decision function.
    '''

    def __init__(
            self,
            n_hidden: Union[List[int], Tuple[int], int] = 64,
            n_layers: int = 3,
            act=nn.ReLU,
            alpha: float = 0.5,
            lr: float = 0.005,
            weight_decay: float = 0.,
            epoch: int = 5,
            verbose: bool = False,
            contamination: float = 0.1,
            num_feature = 2
    ) -> None:
        super().__init__(contamination)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.act = act
        self.alpha = alpha
        self.lr = lr
        self.weight_decay = weight_decay
        self.epoch = epoch
        self.verbose = verbose
        self.model = DOMINANT_MODEL(
            num_feature,
            self.n_hidden,
            self.n_layers,
            self.act,
        )

    def fit(self, G: Data, y=None):

        A = to_dense_adj(G.edge_index, max_num_nodes=G.num_nodes)[0]
        optim = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.model.train()
        for epoch in range(1, self.epoch + 1):
            stru_recon, attr_recon = self.model(G.x, G.edge_index)
            stru_score = torch.square(stru_recon - A).sum(1)
            attr_score = torch.square(attr_recon - G.x).sum(1)
            score = self.alpha * stru_score + (1 - self.alpha) * attr_score
            loss = score.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose:
                log = "Epoch {:3d}, loss={:5.6f}".format(epoch, loss.item())
                if y is not None:
                    auc = roc_auc_score(y, score.detach().numpy())
                    log += ", AUC={:6f}".format(auc)
                print(log)

        self.decision_scores_ = self.decision_function(G)
        self.labels_, self.threshold_ = predict_by_score(
            self.decision_scores_,
            self.contamination,
            True,
        )
        return self

    def predict(self, G: Data):
        ano_score = self.decision_function(G)
        return predict_by_score(ano_score, self.contamination)


def GCN(
    n_layers: int,
    n_input: int,
    n_hidden: Union[int, List[int], Tuple[int]],
    n_output: int,
    act: nn.modules,
    last_act: bool = True,
    conv_layer: nn.Module = GCNConv,
):
    if n_layers < 0:
        raise ValueError("Parameter 'n_layers' must be non-negative!")
    elif n_layers == 0:
        return Sequential('x, edge_index', [(nn.Identity(), 'x -> x')])

    if type(n_hidden) not in {list, tuple}:
        n_hidden = [n_hidden] * max(n_layers - 1, 0)

    n_per_layer = [n_input] + n_hidden + [n_output]
    assert len(n_per_layer) == n_layers + 1
    module_list = []
    for i in range(n_layers):
        module_list.extend([
            (
                conv_layer(n_per_layer[i], n_per_layer[i + 1]),
                'x, edge_index -> x',
            ),
            act(),
        ])

    if not last_act:
        module_list.pop()
    return Sequential('x, edge_index', module_list)


def predict_by_score(
    score: np.ndarray,
    contamination: float,
    return_threshold: bool = False,
):
    pred = np.zeros_like(score)
    #threshold = np.percentile(score, 1 - contamination)
    threshold = np.percentile(score, contamination)

    pred[score>threshold] = 1

    if return_threshold:
        return pred, threshold
    return pred