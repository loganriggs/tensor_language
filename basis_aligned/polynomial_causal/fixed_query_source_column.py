"""Exact conditional source-column update; normalized/rotated Q,K are explicit ports."""
import torch
from attention8h2_rankone_edge_v1 import rankone_write

def source_column(q1,q2,k10,k20,v0,k11,k21,v1,output,positions,cached_writers=None):
    """Q[B,T,H,E], K/V[B,H,E], O[D,H*E]; cached value already mixed.

    Other source tokens and both query fields stay fixed. No source generator
    or baseline context is hidden in this executor. Every head is retained.
    """
    b,t,h,e=q1.shape
    def routing(k1,k2):
        a=torch.einsum('bthe,bhe->bth',q1,k1)/e
        c=torch.einsum('bthe,bhe->bth',q2,k2)/e
        mask=torch.arange(t,device=q1.device)[None]>=positions[:,None]
        return (a*c)*mask[:,:,None]
    a0,a1=routing(k10,k20),routing(k11,k21)
    # a1*v1-a0*v0 = a1*(v1-v0)+(a1-a0)*v0.
    # Old writers can be cached across edits sharing this background.
    writes=[]
    for head in range(h):
        w=output[:,head*e:(head+1)*e]
        old=rankone_write((a1-a0)[:,:,head],v0[:,head],w) if cached_writers is None else (a1-a0)[:,:,head,None]*cached_writers[:,head,None,:]
        writes.append(rankone_write(a1[:,:,head],(v1-v0)[:,head],w)+old)
    return torch.stack(writes).sum(0)

def price(tokens,d=1152,heads=9):
    e=d//heads
    return dict(baseline_query_ports=2*tokens*d,baseline_key_value_ports=3*d,
                new_source_key_value_ports=3*d,static_output_weights=d*d,
                baseline_cached_writers=heads*d,
                edited_writer_projection_multiplies_with_cache=d*d,
                edited_writer_projection_multiplies_without_cache=2*d*d,
                route_dots_multiplies=4*tokens*d,
                two_route_broadcast_multiplies=2*tokens*heads*d,
                dense_per_token_output_projection_multiplies=tokens*d*d,
                excludes='source projections, input/head normalization, baseline queries/writers, suffix and full model; these remain charged dependencies')
