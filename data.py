import os
import torch

def load_data(args):

    FILE_NAME = "{}.pt".format(args.real_world_name)
    FILE_PATH = os.path.join(args.data_dir, FILE_NAME)
    data = torch.load(FILE_PATH)
    anomaly_flag = data.y.numpy()

    return data, anomaly_flag
