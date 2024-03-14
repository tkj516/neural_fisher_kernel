from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import inf
from torch.utils._foreach_utils import (
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

def compute_spectrum(basis_size, basis):
    spectrum = [0] * basis_size
    for k in basis.keys():
        for i in range(basis_size):
            spectrum[i] += torch.linalg.norm(basis[k][i].flatten()).cpu().numpy() ** 2
    return np.sqrt(np.array(spectrum))

def clip_grad_norm_(
    parameters: Dict[str, torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native 
            tensors and silently fall back to the slow implementation for other device 
            types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    grads = parameters
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)

    keys = list(grads.keys())
    first_device = grads[keys[0]].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] = (
        _group_tensors_by_device_and_dtype([[v.detach() for (k, v) in grads.items()]])
    )  # type: ignore[assignment]

    if norm_type == inf:
        norms = [v.detach().abs().max().to(first_device) for (k, v) in grads.items()]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for (device, _), [grads] in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(
                grads, device=device
            ):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(
                    f"foreach=True was passed, but can't use the foreach API on "
                    f"{device.type} tensors"
                )
            else:
                norms.extend([torch.norm(g, norm_type) for g in grads])

        total_norm = torch.norm(
            torch.stack([norm.to(first_device) for norm in norms]), norm_type
        )

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1,
    # but doing so avoids a `if clip_coef < 1:` conditional which can require a 
    # CPU <=> device synchronization when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    clipped_grads = {}
    for k in parameters.keys():
        clipped_grads[k] = (
            parameters[k].detach().mul_(clip_coef_clamped.to(first_device))
        )

    return total_norm