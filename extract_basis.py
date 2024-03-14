"""
https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
"""
import argparse
from types import SimpleNamespace
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.utils import clip_grad_norm_, compute_spectrum
from models.energy_net import energynet
from data_utils import DataProcessing
from fit import map_gradient, fisher_matrix_vector

parser = argparse.ArgumentParser(description='main function of pinciple basis extraction')
parser.add_argument('--dataset_path', type=str, default='/dccstor/mitibm_uq/data/')
parser.add_argument('--root_path', type=str, default='/dccstor/mitibm_uq/neural_fisher_kernel/')
parser.add_argument('--dataset_name', default='CIFAR10', type=str, help='dataset to run')
parser.add_argument('--model_name', default='MobileNet', type=str, help='model to train')
parser.add_argument('--batch_size', type=int, default=32, help="batch_size for training the model")
parser.add_argument('--basis_size', type=int, default=20, help="number of principle basis")
parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
parser.add_argument('--total_iterations', type=int, default=5000, help="number of training iterations")
parser.add_argument('--option', type=str, default = 'multiple' ,choices=['single', 'multiple'])
args = parser.parse_args()

configs_dict = {
    'gradient_clip': 100.0,
}
configs = SimpleNamespace(**configs_dict, **vars(args))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Set up a multiprocessing pool to parallelize computations
    # multiprocessing.set_start_method("spawn")
    # Create a summary writer
    writer = SummaryWriter(log_dir=configs.root_path+'runs/exp')

    data_processing = DataProcessing(configs.dataset_name, root_path=configs.dataset_path, batch_size=configs.batch_size)
    trainset, valset, testset, trainloader, valloader, testloader = data_processing.get_dataloader()

    # Get the model parameters
    if configs.model_name == 'MobileNet':
        model_path = configs.dataset_path+'state_dicts/mobilenet_v2.pt'
    else:
        raise NotImplementedError
    model = energynet(configs.model_name, model_path, pretrained=True).to(DEVICE)
    model.eval()
    saving_path = os.path.join(
        configs.root_path+'checkpoints',
        f'model_{configs.model_name}'
        f'_dataset_{configs.dataset_name}'
        f'_basis_size_{configs.basis_size}'
        f'_option_{configs.option}'
    )

    # Create the basis
    params = dict(model.named_parameters())
    basis = {}
    for k in params.keys():
        basis[k] = torch.rand(configs.basis_size, *params[k].shape, device=DEVICE)

    # pool = multiprocessing.Pool()
    step = 0
    while True:
        for i, samples in tqdm(enumerate(trainloader)):
            samples = samples[0].to(DEVICE)
            vjp_outs = fisher_matrix_vector(configs, model, params, basis, samples)
            gradient = dict(map(map_gradient, zip(vjp_outs.items(), basis.items())))
            clip_grad_norm_(gradient, configs.gradient_clip)
            with torch.no_grad():
                for k in basis.keys():
                    if torch.isnan(gradient[k]).any():
                        raise ValueError(f"NaNs encountered in gradient[{k}]")
                    new_val = basis[k] - configs.lr * gradient[k]
                    basis[k].copy_(new_val)

            if step % 50 == 0:
                fig, ax = plt.subplots()
                spectrum = compute_spectrum(configs.basis_size, basis)
                ax.plot(compute_spectrum(configs.basis_size, basis))
                writer.add_figure("spectrum", fig, step)

            if step % 5000 == 0:
                if not os.path.exists(saving_path):
                    os.makedirs(saving_path)
                torch.save(basis, os.path.join(saving_path, f'basis_{step}.pt'))
            if step >= configs.total_iterations:
                exit(0)
            step += 1
    # pool.close()
