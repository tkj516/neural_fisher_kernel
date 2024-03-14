import argparse
from types import SimpleNamespace
import functools
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.finetune_net import finetunenet
from data_utils import DataProcessing
from train_finetune import train_finetune

parser = argparse.ArgumentParser(description='main function of model adaptation')
parser.add_argument('--dataset_path', type=str, default='/dccstor/mitibm_uq/data/')
parser.add_argument('--root_path', type=str, default='/dccstor/mitibm_uq/neural_fisher_kernel/')
parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='dataset to run')
parser.add_argument('--finetune_dataset_name', default='SVHN', type=str, help='target dataset model adapt to')
parser.add_argument('--model_name', default='MobileNet', type=str, help='model to train')
parser.add_argument('--batch_size', type=int, default=32, help="batch_size for training the model")
parser.add_argument('--basis_size', type=int, default=20, help="number of principle basis")
parser.add_argument('--basis_step', type=int, default=40000, help="which checkpoint of basis to load")
parser.add_argument('--option', type=str, default = 'multiple' ,choices=['single', 'multiple'])
parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
parser.add_argument('--total_iterations', type=int, default=5000, help="number of training iterations")
parser.add_argument('--saving_frequency', type=int, default=5)
args = parser.parse_args()

configs_dict = {

}
configs = SimpleNamespace(**configs_dict, **vars(args))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Set up a multiprocessing pool to parallelize computations
    # multiprocessing.set_start_method("spawn")
    # Create a summary writer
    writer = SummaryWriter(log_dir=configs.root_path+'runs/exp')

    data_processing = DataProcessing(configs.finetune_dataset_name, root_path=configs.dataset_path, batch_size=configs.batch_size)
    trainset, valset, testset, trainloader, valloader, testloader = data_processing.get_dataloader()

    # Get the model parameters
    if configs.model_name == 'MobileNet':
        model_path = configs.dataset_path+'state_dicts/mobilenet_v2.pt'
    else:
        raise NotImplementedError
    basis_path = os.path.join(
        configs.root_path + 'checkpoints',
        f'model_{configs.model_name}'
        f'_dataset_{configs.dataset_name}'
        f'_basis_size_{configs.basis_size}'
        f'_option_{configs.option}'
    )
    basis_path = os.path.join(basis_path, f'basis_{configs.basis_step}.pt')
    model = finetunenet(configs, model_path, basis_path, DEVICE).to(DEVICE)
    model.train()

    saving_path = os.path.join(
        configs.root_path + 'checkpoints_finetune',
        f'model_{configs.model_name}'
        f'_source_{configs.finetune_dataset_name}'
        f'_target_dataset_{configs.finetune_dataset_name}'
        f'_basis_size_{configs.basis_size}'
    )
    train_finetune(configs, model, trainloader, valloader, saving_path, DEVICE)



