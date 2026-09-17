"""Exact conditional odd-value write without inherited-first or initial-state inputs.

The raw mixed block9 residual is supplied directly. It is a native-state port,
not generated here. Upstream head8 delta stays explicit; full odd routing keeps
both QK factors and native normalizers. Native replay is a separate requirement.
"""
import torch
import torch.nn.functional as F
from odd_source_swap_interaction_v1 import source_factors
from odd_source_positions_v1 import select_sources


def execute(graph, raw_mixed9, delta8, lambda90, destination):
    current=F.rms_norm(raw_mixed9,(raw_mixed9.shape[-1],))
    changed=F.rms_norm(raw_mixed9+lambda90*delta8,(raw_mixed9.shape[-1],))
    unused=torch.zeros((*current.shape[:2],128),device=current.device,dtype=current.dtype)
    routing,_=source_factors(graph,current,current,current,unused)
    # The inherited value is identical in both arms and cancels exactly.
    value_delta=(1-graph.p['mixture'].double())*((changed.double()-current.double())@graph.p['current_value'].double().T)
    mask=torch.as_tensor(destination,device=current.device)
    if mask.ndim==1:mask=mask[None].expand(current.shape[0],-1)
    return select_sources(routing*value_delta[:,None],mask)@graph.p['output'].double().T
