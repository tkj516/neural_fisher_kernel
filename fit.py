import functools
import numpy as np
import torch
import torch.nn.functional as F


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

@torch.no_grad()
def fisher_matrix_vector(configs, model, params, basis, samples):
    # Create a function that applies the model to a single input with the given
    # parameters
    def _model_fn(parameters, input):
        return torch.func.functional_call(model, parameters, (input.unsqueeze(0),)).squeeze(0)
    # First we need to compute U_x @ V, where U_x = u(x) - mean(u(x)). Here we define
    # u(x): = \sum_{y \in Y} p(y | x) \nabla_{\th} f^y(x).  The product with V can be
    # computed as a JVP.
    if configs.option == 'single':
        # Define the JVP w.r.t a single cotangent
        @functools.partial(torch.func.vmap, in_dims=(None, 0, None))
        def _jvp(parameters, inputs, cotangents):
            def _inputs_forward(_parameters):
                return _model_fn(_parameters, inputs)
            return torch.func.jvp(_inputs_forward,(parameters,),(cotangents,),)

        # Need to computes JVP with basis manually per basis vector as we cannot VMAP
        # over a dictionary
        jvp_outs = []
        for basis_num in range(configs.basis_size):
            basis_l = {}
            for k in basis.keys():
                basis_l[k] = basis[k][basis_num]

            _, jvp_out = _jvp(params, samples, basis_l)

            jvp_outs.append(jvp_out)
        jvp_outs = torch.stack(jvp_outs, dim=0)  # (BASIS_SIZE, C)
    elif configs.option == 'multiple':
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
    else:
        raise NotImplementedError

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
            return torch.func.functional_call(model, _parameters, (inputs.unsqueeze(0),)).squeeze(0)
        _, vjp_fn = torch.func.vjp(_scales_fn, parameters)
        return vjp_fn(cotangents)[0]

    cotangents = torch.concat([u, torch.mean(mean, dim=1, keepdim=True).expand((-1, configs.batch_size // 2))], dim=1)  # (BASIS_SIZE, B)
    vjp_outs = _vjp(params, samples, cotangents)  # (BASIS_SIZE, ...)

    return vjp_outs