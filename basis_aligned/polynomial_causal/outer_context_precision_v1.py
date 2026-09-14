"""Execute OUTER_CONTEXT_PRECISION_V1_PREREGISTRATION.md without adaptive extension."""
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import head17_output_block_objective_v1 as builder
from normalized_pair_ht_control_20260914_0256 import CHECKPOINT
from retained_contraction_error_control_v1 import components
from extracted_circuits.three_corner_head17_interaction_v1.ports import EPS, project, from_projections, additive

P = Path(__file__).resolve().parent


@torch.no_grad()
def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    signal.alarm(120)
    started = time.perf_counter()
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    builder.CHECKPOINT = CHECKPOINT
    tensor, ids = builder.build()
    path = P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt'
    package = torch.load(path, weights_only=True, map_location='cpu')
    mask = torch.from_numpy(np.unpackbits(package['mask'].numpy(), bitorder='little', count=tensor.numel()).copy()).bool()
    flat = torch.zeros(tensor.numel(), dtype=torch.float64)
    flat[mask] = package['values'].double()
    error = torch.einsum('op,pih,ah->oia', package['output'].double(), flat.reshape(package['shape']), package['head'].double())-tensor
    writer = torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt', weights_only=True)['output_matrix'].double()
    def head(layer, name):
        return sd[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9,128,1152)[2].double()
    maps = tuple(head(17, k) for k in ('c_q','c_k','c_q2','c_k2','c_v'))
    first_map = head(0, 'c_v')
    left, right, down = (sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down'))
    bias = sd['transformer.h.17.mlp.Down_bias'].double()
    mixture = float(sd['transformer.h.17.attn.lamb'])
    grams, energies, references = [], [], []
    replay = 0.
    for seed in range(170215000, 170215512):
        generator = torch.Generator().manual_seed(seed)
        def rand():
            return torch.randn(64,5,1152,generator=generator,dtype=torch.float64)
        x, dc, dr, initial = rand(), .1*rand(), .2*rand(), rand()
        corners = (x,x+dc,x+dr)
        projections = tuple(project(c,maps) for c in corners)
        norms = tuple(c.square().mean(-1)+EPS for c in corners)
        added_norm = norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
        first = (initial/(initial.square().mean(-1)+EPS).sqrt()[...,None])@first_map.T
        ports = tuple(from_projections(p,r,first,mixture) for p,r in zip(projections,norms))
        ports += (from_projections(additive(*projections),added_norm,first,mixture),)
        terms = components(*ports)
        z = x[:,-1]
        state = z+terms.sum(1)@writer.T
        r2 = state.square().mean(-1)+EPS
        normalized = state/r2.sqrt()[:,None]
        final = state+((normalized@left.T)*(normalized@right.T))@down.T+bias
        denominator = r2*(final.square().mean(-1)+EPS).sqrt()
        branches = torch.einsum('oih,ni,nkh->nko',error,z,terms)/denominator[:,None,None]
        gram = torch.einsum('nko,nlo->nkl',branches,branches)
        energy = branches.sum(1).square().sum(-1)
        reference = torch.einsum('oih,ni,nh->no',tensor,z,terms.sum(1))/denominator[:,None]
        replay = max(replay,float((gram.sum((1,2))-energy).norm()/energy.norm()))
        grams.append(gram); energies.append(energy); references.append(reference.square().sum(-1))
    energy = torch.cat(energies)
    gram = torch.cat(grams)
    mean = float(energy.mean())
    se = float(energy.std()/len(energy)**.5)
    halfgap = float(abs(energy[:16384].mean()-energy[16384:].mean())/energy.mean())
    changes = [float(abs(energy[:n].mean()-energy[:n//2].mean())/energy[:n].mean()) for n in (16384,32768)]
    result = dict(utc=datetime.now(timezone.utc).isoformat(), contexts=len(energy),
        seeds=[170215000,170215511], target_token_ids=ids, mean_energy=mean,
        relative_standard_error=se/mean, independent_half_relative_gap=halfgap,
        panel_relative_standard_error=float(energy.reshape(512,64).mean(1).std()/512**.5/mean),
        doubling_relative_changes=changes, mean_gram=gram.mean(0).tolist(),
        gram_entry_standard_errors=(gram.std(0)/len(energy)**.5).tolist(),
        relative_mixed_error_ratio_of_means=float((energy.mean()/torch.cat(references).mean()).sqrt()),
        replay_relative_error=replay, pred_a=replay<=1e-10 and bool(torch.isfinite(gram).all()),
        pred_b=se/mean<=.01 and halfgap<=.01, pred_c=max(changes)<=.01,
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seconds=time.perf_counter()-started,
        scope='Frozen native weights, synthetic Gaussian raw corners, exact source sums/all nine moments and conditional final RMS. No fit, native-text fidelity, softcap, or finite-sample statistical guarantee.')
    signal.alarm(0)
    return result


if __name__ == '__main__':
    target = P/'OUTER_CONTEXT_PRECISION_V1_RESULT.json'
    assert not target.exists()
    result = run()
    with target.open('x') as handle:
        json.dump(result,handle,indent=2)
        handle.write('\n')
    print(json.dumps(result,indent=2))
