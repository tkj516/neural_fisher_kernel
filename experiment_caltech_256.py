"""
https://pytorch.org/functorch/stable/notebooks/neural_tangent_kernels.html
"""

import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import Caltech256
from tqdm import tqdm
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    MobileNetV3,
)
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
from utils.clip_grad_norm import clip_grad_norm_

BATCH_SIZE = 32
DATA_DIR = "/home/tejasj/data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASIS_SIZE = 10
LEARNING_RATE = 0.01
TOTAL_ITERS = 50_000


class EnergyNet(MobileNetV3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns \sum_{y \in Y} p(y | x) f^y(x).
        """
        out = super().forward(x)
        probs = F.softmax(out, dim=-1)
        return torch.sum(probs.detach() * out, dim=-1)


def energynet():
    original_model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2
    )
    state_dict = original_model.state_dict()
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large")
    model = EnergyNet(
        inverted_residual_setting=inverted_residual_setting, last_channel=last_channel
    )
    model.load_state_dict(state_dict)
    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    return model


@torch.no_grad()
def fisher_matrix_vector(model, params, basis, samples, option=2):
    # Create a function that applies the model to a single input with the given
    # parameters
    def _model_fn(parameters, input):
        return torch.func.functional_call(
            model, parameters, (input.unsqueeze(0),)
        ).squeeze(0)

    # First we need to compute U_x @ V, where U_x = u(x) - mean(u(x)). Here we define
    # u(x): = \sum_{y \in Y} p(y | x) \nabla_{\th} f^y(x).  The product with V can be
    # computed as a JVP.
    if option == 1:
        # Define the JVP w.r.t a single cotangent
        @functools.partial(torch.func.vmap, in_dims=(None, 0, None))
        def _jvp(parameters, inputs, cotangents):
            def _inputs_forward(_parameters):
                return _model_fn(_parameters, inputs)

            return torch.func.jvp(
                _inputs_forward,
                (parameters,),
                (cotangents,),
            )

        # Need to computes JVP with basis manually per basis vector as we cannot VMAP
        # over a dictionary
        jvp_outs = []
        for basis_num in range(BASIS_SIZE):
            basis_l = {}
            for k in basis.keys():
                basis_l[k] = basis[k][basis_num]

            _, jvp_out = _jvp(params, samples, basis_l)

            jvp_outs.append(jvp_out)
        jvp_outs = torch.stack(jvp_outs, dim=0)  # (BASIS_SIZE, C)
    elif option == 2:
        # Define the JVP w.r.t multiple cotangents using jacrev at the cost of
        # potentially higher memory usage
        @functools.partial(torch.func.vmap, in_dims=(None, 0, None), out_dims=1)
        def _jvp(parameters, inputs, cotangents):
            jac = torch.func.jacrev(_model_fn, argnums=0)(parameters, inputs)

            # Carry out the JVP manually
            jvp_out = 0
            for k in jac.keys():
                jvp_out += torch.einsum("...,l...->l", jac[k], cotangents[k])

            return jvp_out

        if torch.isnan(samples).any():
            raise ValueError("NaNs encountered in samples")
        for k in basis.keys():
            if torch.isnan(basis[k]).any():
                raise ValueError(f"NaNs encountered in basis[{k}]")

        jvp_outs = _jvp(params, samples, basis)

    if torch.isnan(jvp_outs).any():
        raise ValueError("NaNs encountered in JVP")

    # This computes u(x)^T @ V and \mu^T @ V
    u, mean = torch.chunk(jvp_outs.detach(), 2, dim=1)  # (BASIS_SIZE, B // 2)

    # We ultimately need to compute \E [(U_x @ U_x^T) @ V], which can be done by solving
    # \E [(u(x) - \mu) @ (u(x) - \mu)^T @ V]. Note that this is equivalent to computing
    # \E [u(x) @ u(x)^T @ V] - \mu @ \mu^T @ V.  We have already computed u(x)^T @ V and
    # \mu^T @ V with the above JVP.
    #
    # Let's focus on a single term for a fixed x: The term u(x) @ u(x)^T @ V can be
    # computed using a VJP now.  Now the cotangent is u(x)^T @ V and we still forward
    # through the same function but calculate the tranposed jacobian implicitly. Again
    # we use the first half of the batch for this.
    #
    # We can do the same thing for \mu @ \mu^T @ V, but now we just reuse the same
    # \mu^T @ V for each sample in the second half of the batch and then take the
    # average.
    @functools.partial(torch.func.vmap, in_dims=(None, None, 0))
    @functools.partial(torch.func.vmap, in_dims=(None, 0, 0))
    def _vjp(parameters, inputs, cotangents):
        def _scales_fn(_parameters):
            return torch.func.functional_call(
                model, _parameters, (inputs.unsqueeze(0),)
            ).squeeze(0)

        _, vjp_fn = torch.func.vjp(_scales_fn, parameters)
        return vjp_fn(cotangents)[0]

    cotangents = torch.concat(
        [u, torch.mean(mean, dim=1, keepdim=True).expand((-1, BATCH_SIZE // 2))], dim=1
    )  # (BASIS_SIZE, B)
    vjp_outs = _vjp(params, samples, cotangents)  # (BASIS_SIZE, ...)

    return vjp_outs


def map_gradient(item):
    item_vjp, item_basis = item
    key_vjp, vjp = item_vjp
    key_basis, basis = item_basis
    assert key_vjp == key_basis, f"Keys do not match: {key_vjp} != {key_basis}"

    # This computes U_x @ U_x^T @ V
    u2, mean = torch.chunk(vjp.detach(), 2, dim=1)
    # This computes V @ V^T @ V
    w = basis.flatten(start_dim=1)
    wTw = torch.triu(torch.einsum("ld,Ld->lL", w, w), diagonal=1)

    if torch.isnan(wTw).any():
        raise ValueError("NaNs encountered in wTw")

    return (
        key_vjp,
        -torch.mean(u2 - torch.mean(mean, dim=1, keepdim=True), dim=1)
        + torch.einsum("l...,lL->L...", basis, wTw),
    )


def compute_spectrum(basis):
    spectrum = [0] * BASIS_SIZE
    for k in basis.keys():
        for i in range(BASIS_SIZE):
            spectrum[i] += torch.linalg.norm(basis[k][i].flatten()).cpu().numpy() ** 2

    return np.sqrt(np.array(spectrum))


if __name__ == "__main__":
    # Set up a multiprocessing pool to parallelize computations
    # multiprocessing.set_start_method("spawn")

    # Create a summary writer
    writer = SummaryWriter(
        log_dir="/home/tejasj/data/neural_fisher_kernel/runs/exp_caltech_256"
    )

    transform = MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()
    dataset = Caltech256(root=DATA_DIR, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
    )

    # Get the model parameters
    model = energynet().to(DEVICE)
    model.eval()
    params = dict(model.named_parameters())

    # Create the basis
    basis = {}
    for k in params.keys():
        basis[k] = torch.rand(BASIS_SIZE, *params[k].shape, device=DEVICE)

    # pool = multiprocessing.Pool()

    step = 0

    while True:
        for i, samples in tqdm(enumerate(dataloader)):
            samples = samples[0].to(DEVICE)
            vjp_outs = fisher_matrix_vector(model, params, basis, samples)
            gradient = dict(map(map_gradient, zip(vjp_outs.items(), basis.items())))
            clip_grad_norm_(gradient, 100.0)

            with torch.no_grad():
                for k in basis.keys():
                    if torch.isnan(gradient[k]).any():
                        raise ValueError(f"NaNs encountered in gradient[{k}]")
                    new_val = basis[k] - LEARNING_RATE * gradient[k]
                    basis[k].copy_(new_val)

            if step % 50 == 0:
                fig, ax = plt.subplots()
                spectrum = compute_spectrum(basis)
                ax.plot(compute_spectrum(basis))
                writer.add_figure("spectrum", fig, step)

            if step % 5000 == 0:
                if not os.path.exists(
                    "/home/tejasj/data/neural_fisher_kernel/checkpoints/mobilenet_v3_l_20"
                ):
                    os.makedirs(
                        "/home/tejasj/data/neural_fisher_kernel/checkpoints/mobilenet_v3_l_20"
                    )
                torch.save(
                    basis,
                    f"/home/tejasj/data/neural_fisher_kernel/checkpoints/mobilenet_v3_l_20/basis_{step}.pt",
                )

            if step >= TOTAL_ITERS:
                exit(0)

            step += 1

    # pool.close()
