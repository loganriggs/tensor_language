"""Model-free helpers for grouped regional bilinear interaction runners."""
from __future__ import annotations

import torch


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint rotary convention to [B,T,D] or [B,T,H,D]."""
    if x.ndim not in (3, 4) or cos.ndim != 4 or sin.shape != cos.shape:
        raise ValueError("expected x [B,T,D] or [B,T,H,D] and cos/sin [1,T,1,D/2]")
    if x.shape[1] != cos.shape[1] or x.shape[-1] != 2*cos.shape[-1]:
        raise ValueError("rotary length or width mismatch")
    phase_cos=(cos if x.ndim==4 else cos[:,:,0]).to(x.dtype)
    phase_sin=(sin if x.ndim==4 else sin[:,:,0]).to(x.dtype)
    d=x.shape[-1]//2; first,second=x[...,:d],x[...,d:]
    return torch.cat((first*phase_cos+second*phase_sin,
                      first*(-phase_sin)+second*phase_cos),dim=-1)


def paired_difference(x: torch.Tensor, row_axis: int) -> torch.Tensor:
    """Subtract odd rows from preceding even rows along an explicit axis."""
    axis=row_axis % x.ndim
    if x.shape[axis] % 2: raise ValueError("paired row axis must have even length")
    even=[slice(None)]*x.ndim; odd=[slice(None)]*x.ndim
    even[axis]=slice(0,None,2); odd[axis]=slice(1,None,2)
    return x[tuple(even)]-x[tuple(odd)]


def split_named_sources(names, tensors, selected_names):
    """Return exact selected and remainder sums after validating the partition."""
    if len(names)!=len(tensors) or len(set(names))!=len(names): raise ValueError("source names/tensors mismatch")
    selected=set(selected_names)
    if not selected or not selected.issubset(names): raise ValueError("selected source set is empty or unknown")
    chosen=sum(t for n,t in zip(names,tensors) if n in selected)
    remainder=sum(t for n,t in zip(names,tensors) if n not in selected)
    return chosen,remainder


def ordered_bilinear_scores(query_groups, key_groups, width: int):
    """Return [DD, DR, RD, RR] score tensors for two projected groups."""
    if len(query_groups)!=2 or len(key_groups)!=2: raise ValueError("exactly two query/key groups required")
    return torch.stack([torch.einsum("bqd,bkd->bqk",q,k)/width
                        for q in query_groups for k in key_groups])
