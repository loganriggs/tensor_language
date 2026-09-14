"""Eliminate separate basis reads after folding the directional producer."""
import torch
from shared_key_reads_v1 import compile_program as shared_compile
from contracted_qk_response_v1 import features, scalar as old_scalar


def compile_program(native,low,gain):
    p=shared_compile(native,low,gain)
    k=torch.cat([native['k1'][1],native['k2'][1]]).double()
    p['key_coordinates']=torch.linalg.lstsq(k.T,native['key_basis'][1].double()).solution
    p['readers']=torch.cat([p['readers'][:512],p['readers'][-1:]],0)
    p['read_left']=p['readers']@p['left']
    p['read_direction']=p['readers']@p['direction']
    return p


def expand(reads,p):
    keys=torch.cat([reads[...,128:256],reads[...,384:512]],-1)
    shared=keys@p['key_coordinates']
    return torch.cat([reads[...,:512],shared@p['inside_adapters'][0].T,
                      shared@p['inside_adapters'][1].T,reads[...,-1:]],-1)


def scalar(reads,rho2,p):
    return old_scalar(expand(reads,p),rho2)
