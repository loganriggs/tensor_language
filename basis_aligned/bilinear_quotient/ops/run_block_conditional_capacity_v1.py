#!/usr/bin/env python3
"""pred_a rank4/descent<=1e-8; pred_b one rank8 gain>=1e-4; pred_c >=8blocks.

BQGATE:0forwards0seq. Full-U conditional spectra; no joint fit. Alpha=.0025.
Null: these fixed inputframes hide no appreciable rank8 output expansion.
Prices: each added output adds1152+136coefficients; existing377344 retained.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from conditional_block_svd_v1 import native_targets,conditional_residual,local_cost
from block_conditional_capacity_v1 import spectrum_solution
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/'BLOCK_CONDITIONAL_CAPACITY_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,
                             ranks=[4,8,16,32,64,128],alpha=.0025)));return
    out=P/'BLOCK_CONDITIONAL_CAPACITY_V1_RESULT.json';assert not out.exists()
    signal.alarm(180);torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    prior=json.loads((P/'CONDITIONAL_BLOCK_SVD_V1_RESULT.json').read_text())
    saved=torch.load(prior['cache']['path'],map_location='cpu',weights_only=True)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();chol=torch.linalg.cholesky(u.T@u);del u
    l,r,down=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    down=chol.T@down;bank=saved['bank'].double().cuda()
    writers=(chol.T@saved['writer'].double().cuda()).reshape(1152,16,4).permute(1,0,2)
    cores=saved['packed'].double().cuda().reshape(136,16,4).permute(1,0,2)
    total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    targets=native_targets(l,r,down,bank);rows=[];errors=[]
    for g in range(16):
        residual=conditional_residual(targets,bank,writers,cores,g)
        s=torch.linalg.svdvals(residual)
        old=float(local_cost(residual,writers[g],cores[g],.01))
        old_residual=float(local_cost(residual,writers[g],cores[g],0.))
        capacities=[]
        for rank in (4,8,16,32,64,128):
            solution=spectrum_solution(s,rank)
            capacities.append(dict(rank=rank,active=solution['active'],
                objective_decrease=(old-solution['loss'])/total,
                capture_gain=(old_residual-solution['residual'])/total,
                added_coefficients=(rank-4)*1288,component_penalty=rank*.0025))
        errors.append(abs(capacities[0]['objective_decrease']))
        errors.extend(max(0.,a['objective_decrease']-b['objective_decrease']) for a,b in zip(capacities,capacities[1:]))
        rows.append(dict(block=g,singular_values=s.tolist(),capacities=capacities))
    hits=sum(row['capacities'][1]['capture_gain']>=1e-4 for row in rows)
    result=dict(predictions={'pred_a_instrument':max(errors)<=1e-8,
                            'pred_b_one_rank8_gain':hits>=1,'pred_c_eight_rank8_gains':hits>=8},
                rank8_hit_count=hits,maximum_instrument_error=max(errors),blocks=rows,
                alpha=.0025,wall_seconds=time.perf_counter()-start,binding=binding,
                body_forwards=0,corpus_access=False,
                scope='Isolated expansions with fixed input frames and fixed other blocks; gains cannot be summed. Larger ranks carry increased storage and rescaled component penalty to hold nuclear coefficient fixed. Not a fitted new candidate or circuit evidence.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('binding','blocks')},indent=2),flush=True)
    assert result['predictions']['pred_a_instrument']


if __name__=='__main__':main()
