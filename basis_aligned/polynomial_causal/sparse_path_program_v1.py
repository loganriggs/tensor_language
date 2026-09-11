"""Executable full-output sparse path program with explicit intervention semantics.
No bias/direct residual/final tail is included. Caller supplies actual input RMS squared.
Deleting internal nodes/edges leaves that denominator fixed; swapping sources requires recomputing it.
"""
import math
import torch


def edges(program):
    bank=program['bank'];rank=bank.shape[-1];device=bank.device
    if program['mode']=='joint':
        assert bank.shape[0]==2
        pairs=torch.triu_indices(2*rank,2*rank,device=device)
    else:
        assert program['mode']=='independent' and bank.shape[0]==4
        tri=torch.triu_indices(rank,rank,device=device)
        a,b=torch.meshgrid(torch.arange(rank,device=device),torch.arange(rank,device=device),indexing='ij')
        mixed=torch.stack([a.flatten()+rank,b.flatten()+2*rank])
        pairs=torch.cat([tri,mixed,tri+3*rank],1)
    return pairs[:,program['support']]


def run(program,residual,head_values,input_rms_squared,zero_nodes=(),zero_edges=()):
    bank=program['bank'];rank=bank.shape[-1]
    assert residual.shape==head_values.shape and residual.shape[-1]==bank.shape[-2]
    assert input_rms_squared.shape==residual.shape[:-1] and bool((input_rms_squared>0).all())
    source=[residual,head_values*program['source_scale']]
    assignments=[0,1] if program['mode']=='joint' else [0,0,1,1]
    reads=torch.cat([source[port]@bank[i] for i,port in enumerate(assignments)],-1)
    if zero_nodes:
        assert min(zero_nodes)>=0 and max(zero_nodes)<reads.shape[-1]
        reads=reads.clone();reads[...,list(zero_nodes)]=0
    pair=edges(program);mult=torch.where(pair[0]==pair[1],1.,math.sqrt(2)).to(reads)
    amplitude=reads[...,pair[0]]*reads[...,pair[1]]*mult
    if zero_edges:
        assert min(zero_edges)>=0 and max(zero_edges)<amplitude.shape[-1]
        amplitude=amplitude.clone();amplitude[...,list(zero_edges)]=0
    numerator=amplitude@program['physical_writer'].T
    return dict(reads=reads,amplitudes=amplitude,numerator=numerator,write=numerator/input_rms_squared[...,None])
