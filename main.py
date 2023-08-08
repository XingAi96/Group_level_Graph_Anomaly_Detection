import argparse
import os
import torch
from utils import *
from training import train, test
from initializer import *

def load_data(args):

    FILE_NAME = "{}.pt".format(args.real_world_name)
    FILE_PATH = os.path.join(args.data_dir, FILE_NAME)
    data = torch.load(FILE_PATH)
    anomaly_flag = data.y.numpy()

    return data, anomaly_flag

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--real_world_name', type=str, default='AMLpublic')#simML AMLpublic Ethereum_TSGN Citeseer Cora
    parser.add_argument('--dataset', type=str, default='complete')
    parser.add_argument('--anomaly_type', type=str, default='complete')
    parser.add_argument('--size', type=int, default=200)
    parser.add_argument('--anomaly_ratio', type=float, default=0.02)
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--anomaly_scale', type=float, default=0.3)
    parser.add_argument('--anomaly_attr_ratio', type=float, default=1.0)
    parser.add_argument('--diff_ratio', type=int, default=2)
    parser.add_argument('--half_num', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--num_anchors', type=int, default=0)

    parser.add_argument('--gae_type', type=str, default='dominant')#dominant#subGAE
    parser.add_argument('--gae_embedding_channels', type=int, default=64)
    parser.add_argument('--gae_hidden_channels', type=int, default=16)
    parser.add_argument('--gae_num_layer', type=int, default=2)
    parser.add_argument('--gae_out_channels', type=int, default=16)
    parser.add_argument('--gae_num_features', type=int, default=2)
    parser.add_argument('--gae_epochs', type=int, default=100)
    parser.add_argument('--gae_lr', type=float, default=1e-2)

    parser.add_argument('--gcl_input_dim', type=int, default=1)
    parser.add_argument('--gcl_hidden_dim', type=int, default=32)
    parser.add_argument('--gcl_num_layer', type=int, default=2)
    parser.add_argument('--gcl_output_dim', type=int, default=16)
    parser.add_argument('--gcl_epochs', type=int, default=100)
    parser.add_argument('--gcl_lr', type=float, default=1e-2)
    parser.add_argument('--q', type=int, default=90)
    parser.add_argument('--inner_epochs', type=int, default=20)
    parser.add_argument('--inner_lr', type=float, default=1e-4)
    parser.add_argument('--contamination', type=float, default=0.15)

    parser.add_argument('--warmup', type=int, default=90)
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--beta_1', type=float, default=1.)
    parser.add_argument('--beta_2', type=float, default=1.)
    parser.add_argument('--per_ratio', type=float, default=.0)
    parser.add_argument('--convergence', type=float, default=1e-4)
    parser.add_argument('--ending_rounds', type=int, default=1)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load and preprocess data
    data, anomaly_flag = load_data(args)
    batch = set(data.batch.cpu().numpy())
    data = data.to(device)
    data.A, data.G = preprocess_data(data, 's')

    # prepar model parameters
    args.gae_out_channels = data.num_features
    args.gae_num_features = data.num_features
    args.gcl_input_dim = data.num_features

    args.aug1, args.aug2 = A.SubIncreasing(),A.SubDecreasing()
    f1, auc, cratio, rration, cr_list = [], [], [], [], []
    for i in range(1, 11):
        bat = data.batch.cpu().numpy().squeeze()
        y = data.y.cpu().numpy()
        ano_node = 0
        for k in bat:
            if y[k] == 1:
                ano_node += 1
        GAEmodel, GCLmodel, batch_list = train(args, data)
        test_result = test(args, GAEmodel, GCLmodel, data)

        print(f'(Epoch-{i}): Best test F1={test_result["f1"]:.4f},'
              f' AUC={test_result["auc"]:.4f}, CR={test_result["cr"]:.4f}')
        f1.append(test_result["f1"])
        auc.append(test_result["auc"])
        cr_list.append(test_result["cr"])

    f1 = np.array(f1)
    cr_list = np.array(cr_list)
    print(
        f'Average test F1={np.mean(f1):.4f}, AUC={np.mean(auc):.4f}, CR={np.mean(cr_list):.4f}')
    print(
        f'Std test F1={np.std(f1):.3f}, AUC={np.std(auc):.3f}, CR={np.std(cr_list):.3f}')


