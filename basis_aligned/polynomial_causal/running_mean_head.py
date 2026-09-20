"""Extracted signed causal mean at the mixed-value port, no Q/K or model access."""
import torch


def execute(values, diagonal=0., mass=1.):
    """values[B,T,D]; y_t=diagonal*v_t-mass*mean(v_s for s<t).

    The empty prefix at t=0 contributes zero. Scalar diagonal/mass can be
    frozen constants; a tensor diagonal is an explicit, separately priced port.
    """
    count=torch.arange(values.shape[1],device=values.device,dtype=values.dtype).clamp_min(1)
    previous=(values.cumsum(1)-values)/count[None,:,None]
    if torch.is_tensor(diagonal) and diagonal.ndim==2:diagonal=diagonal[...,None]
    return diagonal*values-mass*previous


def execute_projected(normalized_current, normalized_first, current_value,
                      first_value, output, mixture, diagonal=0., mass=1.):
    """Fold two value sources into the mean program and apply its output writer.

    Weight shapes are [head_dim,residual_dim], [head_dim,residual_dim],
    [residual_dim,head_dim]. Upstream normalization is an explicit input boundary.
    """
    values=(1-mixture)*(normalized_current@current_value.T)+mixture*(normalized_first@first_value.T)
    return execute(values,diagonal,mass)@output.T
