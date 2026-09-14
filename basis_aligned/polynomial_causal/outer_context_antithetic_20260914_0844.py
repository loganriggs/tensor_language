"""Frozen CPU audit: 16x64 independent contexts, five sources, 120s cap.

Same synthetic law and frozen sparse candidate as NATIVE_RETAINED_ERROR_V1.
A: all-nine Gram/direct and even-denominator sign replay <=1e-10.
B: independent-half and 256->512->1024 energy differences all <=1%.
C: sign-paired equal-evaluation variance ratio <1 (opposing null >=1).
No fitting, text, native trajectories or adaptive sample extension.
"""
import os, json, time, signal, hashlib
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
    torch.set_num_interop_threads(2)
    signal.alarm(120)
    start = time.perf_counter()
    utc = lambda: datetime.now(timezone.utc).isoformat()
    phases = {'execution_start': utc()}
    sd = torch.load(CHECKPOINT, weights_only=True, mmap=True, map_location='cpu')
    builder.CHECKPOINT = CHECKPOINT
    T, ids = builder.build()
    path = P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt'
    package = torch.load(path, weights_only=True, map_location='cpu')
    mask = torch.from_numpy(np.unpackbits(package['mask'].numpy(), bitorder='little', count=T.numel()).copy()).bool()
    flat = torch.zeros(T.numel(), dtype=torch.float64)
    flat[mask] = package['values'].double()
    fitted = torch.einsum('op,pih,ah->oia', package['output'].double(), flat.reshape(package['shape']), package['head'].double())
    E = fitted-T
    W = torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt', weights_only=True)['output_matrix'].double()
    def head(layer, name):
        return sd[f'transformer.h.{layer}.attn.{name}.weight'].reshape(9,128,1152)[2].double()
    maps = tuple(head(17,k) for k in ('c_q','c_k','c_q2','c_k2','c_v'))
    V0 = head(0,'c_v')
    L,R,D = (sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down'))
    bias = sd['transformer.h.17.mlp.Down_bias'].double()
    mu = float(sd['transformer.h.17.attn.lamb'])
    def evaluate(x,dc,dr,initial):
        corners = (x,x+dc,x+dr)
        projections = tuple(project(c,maps) for c in corners)
        norms = tuple(c.square().mean(-1)+EPS for c in corners)
        nr = norms[1]+norms[2]-norms[0]+2*(dc*dr).mean(-1)
        first = initial/(initial.square().mean(-1)+EPS).sqrt()[...,None] @ V0.T
        ports = tuple(from_projections(p,r,first,mu) for p,r in zip(projections,norms))
        ports += (from_projections(additive(*projections),nr,first,mu),)
        A = components(*ports)
        z = x[:,-1]
        state = z+A.sum(1)@W.T
        r2 = state.square().mean(-1)+EPS
        normed = state/r2.sqrt()[:,None]
        final = state+(normed@L.T)*(normed@R.T)@D.T+bias
        denom = r2*(final.square().mean(-1)+EPS).sqrt()
        numerator = torch.einsum('oih,ni,nkh->nko',E,z,A)
        branches = numerator/denom[:,None,None]
        grams = torch.einsum('nko,nlo->nkl',branches,branches)
        energy = branches.sum(1).square().sum(-1)
        ref = torch.einsum('oih,ni,nh->no',T,z,A.sum(1))/denom[:,None]
        even_denom = r2*(z.square().mean(-1)+EPS).sqrt()
        even_energy = (numerator.sum(1)/even_denom[:,None]).square().sum(-1)
        return grams,energy,ref.square().sum(-1),even_energy,numerator
    positives, negatives, refs, panels = [],[],[],[]
    parity, replay = 0.,0.
    for seed in range(170214840,170214856):
        generator = torch.Generator().manual_seed(seed)
        rand = lambda: torch.randn(64,5,1152,generator=generator,dtype=torch.float64)
        raw = (rand(),.1*rand(),.2*rand(),rand())
        gp,ep,rp,wp,np_ = evaluate(*raw)
        gm,em,rm,wm,nm = evaluate(*(-x for x in raw))
        parity = max(parity,float((wp-wm).norm()/wp.norm()),float((np_-nm).norm()/np_.norm()))
        replay = max(replay,float((gp.sum((1,2))-ep).norm()/ep.norm()))
        positives.append(gp); negatives.append(gm); refs.append(rp)
        panels.append({'seed':seed,'mean_energy':float(ep.mean()),'mean_gram':gp.mean(0).tolist()})
    phases['validation_start'] = utc()
    gp,gm = torch.cat(positives),torch.cat(negatives)
    ep,em = gp.sum((1,2)),gm.sum((1,2))
    ref = torch.cat(refs)
    pair = (ep+em)/2
    covariance = torch.cov(torch.stack((ep,em)))
    ratio = float(4*pair.var()/covariance.trace())
    halfgap = float(abs(ep[:512].mean()-ep[512:].mean())/ep.mean())
    changes = [float(abs(ep[:n].mean()-ep[:n//2].mean())/ep[:n].mean()) for n in (512,1024)]
    se = float(ep.std()/len(ep)**.5)
    panelmeans = ep.reshape(16,64).mean(1)
    result = dict(utc=utc(),phases=phases,contexts=1024,sources=5,panels=panels,
        threads=torch.get_num_threads(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        seeds=[170214840,170214855],target_token_ids=ids,mixture=mu,
        candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mean_gram=gp.mean(0).tolist(),gram_entry_standard_errors=(gp.std(0)/1024**.5).tolist(),
        mean_energy=float(ep.mean()),energy_standard_error=se,relative_standard_error=se/float(ep.mean()),
        panel_relative_standard_error=float(panelmeans.std()/4/ep.mean()),
        energy_normal_approx_95_interval=[float(ep.mean())-1.96*se,float(ep.mean())+1.96*se],
        independent_half_relative_gap=halfgap,doubling_relative_changes=changes,
        diagonal_to_joint=float(gp.mean(0).trace()/ep.mean()),
        relative_mixed_error_ratio_of_means=float((ep.mean()/ref.mean()).sqrt()),
        estimated_n_for_one_percent_standard_error=int(ep.var()/(.01*ep.mean())**2)+1,
        antithetic_correlation=float(covariance[0,1]/(covariance[0,0]*covariance[1,1]).sqrt()),
        antithetic_equal_evaluation_variance_ratio=ratio,
        antithetic_pair_mean=float(pair.mean()),antithetic_pair_standard_error=float(pair.std()/1024**.5),
        final_rms_sign_relative_energy_change=float((ep-em).norm()/ep.norm()),
        even_denominator_and_numerator_sign_replay=parity,gram_replay=replay,
        pred_a=max(parity,replay)<=1e-10,pred_b=max([halfgap]+changes)<=.01,pred_c=ratio<1,
        seconds=time.perf_counter()-start,
        scope='Frozen native weights, synthetic Gaussian raw corners with independent layer0 raw values; exact conditional last MLP/final RMS, no softcap. All nine moments, no fit/native trajectory/fidelity claim. CI and sample-size projections are estimates, not finite-sample guarantees.')
    signal.alarm(0)
    return result

if __name__ == '__main__':
    result = run()
    target = P/'OUTER_CONTEXT_ANTITHETIC_20260914_0844_RESULT.json'
    with target.open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='panels'},indent=2))
