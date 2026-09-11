#!/usr/bin/env python3
# BQGATE: zero body forwards; 512 native tail rows, cached states only, no fitting.
"""Trace-preserving replay of frozen 48-square suffix program; same physical bars.
A source/projection<=1e-9, original margin replay<=1e-4 nats and finite.
B exact3span transfer>=10%fullgap and>=.05 in each task/direction.
C approx3 effect relative RMS<=.25 and sign agreement>=.9 vs exact3 eachcell.
D absolute replacement meanabs CE<=.05 in EACH A1/A2/P/C.
E approx3 swap meanabs CE<=.05 in EACH P/C. Native background retained.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
import hashlib
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal'
STEM='NATIVE_RELATION_TRACE_PHYSICAL_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files']
    assert all(digest(p)==sha for p,sha in binding.items())
    saved=torch.load(P/'NATIVE_TOKEN_RELATION_FOLD_V1.pt',weights_only=True,map_location='cpu')
    assert saved['labels'][:3]==['add_s','add_es','y_to_ies']
    r=saved['physical_readouts'][:3].double();gram=r@r.T
    writers=torch.linalg.solve(gram,r).T
    projection_error=float((r@writers-torch.eye(3,dtype=torch.float64)).abs().max())
    condition=float(torch.linalg.cond(gram));assert projection_error<=1e-9
    cache=torch.load(P/'FROZEN_BRANCH_MORPHOLOGY_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')
    rows=json.loads((P/'FROZEN_BRANCH_MORPHOLOGY_V1_ROWS.json').read_text())['rows']
    old=json.loads((P/'FROZEN_BRANCH_MORPHOLOGY_V1_RESULT.json').read_text());assert old['pred_a'] and old['pred_b']
    assert digest(P/'FROZEN_BRANCH_MORPHOLOGY_V1_ENDPOINTS.pt')==old['cache_sha256']
    x=cache['ports']['input'].double();native=cache['ports']['native_output'].double()
    exact=native@r.T;approx=[]
    trace_corrections=[]
    for j,c in enumerate(saved['compact'][:3]):
        retained_trace=(c['top16_square_coefficients']*c['top16_square_readers'].square().sum(1)).sum()
        correction=-retained_trace/x.shape[1]
        assert abs(float(retained_trace+correction*x.shape[1]))<=1e-9
        trace_corrections.append(float(correction))
        approx.append((x@c['top16_square_readers'].T).square()@c['top16_square_coefficients']+
            (c['radial']+correction)*x.square().sum(1)+saved['folded_bias'][j])
    approx=torch.stack(approx,1);assert torch.isfinite(approx).all()
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,tail_rows=512,projection_error=projection_error,
            readout_gram_condition=condition,fitting=False)));return
    output=P/(STEM+'_RESULT.json');assert not output.exists()
    started=time.perf_counter();torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    unembedding=weights['lm_head.weight'].float().cuda()
    h=(cache['ports']['pre']+cache['ports']['native_output']).cuda()
    node=torch.load(P/'SHARED_NODE_CANONICAL_BRANCHES_V1_SPECTRAL.pt',weights_only=True,map_location='cpu')
    oldwriters=torch.linalg.solve_triangular(node['output_whitener'],node['nodes'][1]['writers'],upper=True)
    amplitudes=cache['amplitudes'].double()
    names=['base','donor','old_bank','exact1','approx1','exact3','approx3','replace3']
    records=[];errors=[];tail_rows=0
    for i,row in enumerate(rows):
        base,donor=2*i,2*i+1
        de,da=exact[donor]-exact[base],approx[donor]-approx[base]
        oldwrite=oldwriters@(amplitudes[donor]-amplitudes[base])
        writes=[oldwrite,r[0]*de[0],r[0]*da[0],writers@de,writers@da,writers@(approx[base]-exact[base])]
        states=torch.stack([h[base],h[donor]]+[h[base]+v.float().cuda() for v in writes])
        logits=30*torch.tanh(F.linear(F.rms_norm(states,(1152,)),unembedding)/30)
        tail_rows+=len(states)
        target,foil=row['donor_answer_id'],row['donor_foil_id']
        margins=(logits[:,target]-logits[:,foil]).double()
        ce=F.cross_entropy(logits.double(),torch.full((len(states),),row['base_answer_id'],device='cuda'),reduction='none')
        shifts=margins-margins[0];cechanges=ce-ce[0]
        assert old['records'][i]['row_id']==row['row_id']
        errors += [abs(float(shifts[1])-old['records'][i]['native_gap']),
                   abs(float(shifts[2])-old['records'][i]['margin_shifts'][2])]
        records.append(dict(row_id=row['row_id'],family=row['family'],direction=row['direction'],
            native_gap=float(shifts[1]),margin_shifts={k:float(v) for k,v in zip(names,shifts)},
            ce_changes={k:float(v) for k,v in zip(names,cechanges)}))
    assert tail_rows==512
    cells=[]
    for family in ('A1','A2'):
        for direction in ('base_to_suffix','suffix_to_base'):
            local=[row for row in records if row['family']==family and row['direction']==direction]
            e=torch.tensor([row['margin_shifts']['exact3'] for row in local],dtype=torch.float64)
            a=torch.tensor([row['margin_shifts']['approx3'] for row in local],dtype=torch.float64)
            gap=sum(row['native_gap'] for row in local)/len(local)
            error=float((a-e).norm()/e.norm());sign=float((a.sign()==e.sign()).double().mean())
            cells.append(dict(family=family,direction=direction,n=len(local),native_gap=gap,
                mean_effects={name:sum(row['margin_shifts'][name] for row in local)/len(local) for name in names[3:]},
                exact_recovery=float(e.mean())/gap,approx_recovery=float(a.mean())/gap,
                relative_effect_error=error,sign_agreement=sign,
                transfer_held=gap>0 and float(e.mean())>=.05 and float(e.mean())/gap>=.1,
                approximation_held=error<=.25 and sign>=.9))
    replacement={};collateral={}
    for family in ('A1','A2','P','C'):
        local=[row for row in records if row['family']==family]
        replacement[family]=sum(abs(row['ce_changes']['replace3']) for row in local)/len(local)
        collateral[family]=sum(abs(row['ce_changes']['approx3']) for row in local)/len(local)
    valid=max(errors)<=1e-4 and all(torch.isfinite(torch.tensor(list(row['margin_shifts'].values()))).all() for row in records)
    a=bool(valid)
    result={'pred_a':a,'pred_b':a and all(c['transfer_held'] for c in cells),
        'pred_c':a and all(c['approximation_held'] for c in cells),
        'pred_d':a and all(v<=.05 for v in replacement.values()),
        'pred_e':a and all(collateral[f]<=.05 for f in ('P','C')),
        'cells':cells,'replacement_mean_abs_ce':replacement,'swap_mean_abs_ce':collateral,
        'maximum_previous_margin_replay_error':max(errors),'projection_error':projection_error,
        'readout_gram_condition':condition,'trace_radial_corrections':trace_corrections,'records':records,'seconds':time.perf_counter()-started,
        'price':{'body_forwards':0,'tail_rows':tail_rows,'square_products':48,'shared_input_radius_required':True,
                 'native_background_retained':True,'physical_writers_floats':writers.numel()},
        'binding_sha256':digest(P/(STEM+'_BINDING.json')),
        'scope':'Fixed suffix readout span with exact dual physical writers. Approximation uses 16 weight-eigen squares per readout, '
        'preserves native trace by correcting the truncated remainder; bias/native complement/background remain. No fitting, fresh OOD or automatic promotion. '
        'Approximate donor differences and absolute replacement have distinct registered criteria.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
