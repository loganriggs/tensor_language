"""Reusable fixed synthetic measure for normalized retained-interaction objectives."""
import numpy as np
import torch
import head17_output_block_objective_v1 as builder
from normalized_pair_ht_control_20260914_0256 import CHECKPOINT
from retained_contraction_error_control_v1 import components
from extracted_circuits.three_corner_head17_interaction_v1.ports import EPS, project, from_projections, additive
from pathlib import Path

P = Path(__file__).resolve().parent


class Contexts:
    def __init__(self):
        self.sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
        builder.CHECKPOINT = CHECKPOINT
        self.tensor, self.ids = builder.build()
        def head(layer, name):
            return self.sd[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9,128,1152)[2].double()
        self.maps = tuple(head(17,k) for k in ('c_q','c_k','c_q2','c_k2','c_v'))
        self.first_map = head(0,'c_v')
        self.writer = torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt', weights_only=True)['output_matrix'].double()
        self.left, self.right, self.down = (self.sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down'))
        self.bias = self.sd['transformer.h.17.mlp.Down_bias'].double()
        self.mixture = float(self.sd['transformer.h.17.attn.lamb'])

    def sample(self, seed, batch=64):
        generator = torch.Generator().manual_seed(seed)
        def rand():
            return torch.randn(batch,5,1152,generator=generator,dtype=torch.float64)
        x, dc, dr, initial = rand(),.1*rand(),.2*rand(),rand()
        corners = (x,x+dc,x+dr)
        projections = tuple(project(c,self.maps) for c in corners)
        norms = tuple(c.square().mean(-1)+EPS for c in corners)
        added_norm = norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
        first = (initial/(initial.square().mean(-1)+EPS).sqrt()[...,None])@self.first_map.T
        ports = tuple(from_projections(p,r,first,self.mixture) for p,r in zip(projections,norms))
        ports += (from_projections(additive(*projections),added_norm,first,self.mixture),)
        terms = components(*ports)
        z = x[:,-1]
        state = z+terms.sum(1)@self.writer.T
        r2 = state.square().mean(-1)+EPS
        normalized = state/r2.sqrt()[:,None]
        final = state+((normalized@self.left.T)*(normalized@self.right.T))@self.down.T+self.bias
        denominator = r2*(final.square().mean(-1)+EPS).sqrt()
        return z,terms,denominator

    def decode(self, name):
        package = torch.load(P/(name+'_PROGRAM.pt'),weights_only=True,map_location='cpu')
        if 'mask' in package:
            mask = torch.from_numpy(np.unpackbits(package['mask'].numpy(),bitorder='little',count=self.tensor.numel()).copy()).bool()
            flat = torch.zeros(self.tensor.numel(),dtype=torch.float64)
            flat[mask] = package['values'].double()
            return torch.einsum('op,pih,ah->oia',package['output'].double(),flat.reshape(package['shape']),package['head'].double())
        assert package['token_ids'] == self.ids
        rows = torch.einsum('ndr,nr->nd',package['output_bases'].double()[package['groups'].long()],package['codes'].double())
        return rows.reshape(1152,128,12).permute(2,0,1)


def branch_errors(error, z, terms, denominator):
    return torch.einsum('oih,ni,nkh->nko',error,z,terms)/denominator[:,None,None]
