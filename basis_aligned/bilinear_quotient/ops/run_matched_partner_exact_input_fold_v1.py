#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachedendpoints,3quadraticforms;300sec.
"""pred_a FP64 exact fold<=1e-8; pred_b FP32 write<=1e-5 and native fidelity;
pred_c counted1,994,688 joint floats and >=4x native-factor byte reduction.
Conditional x16/denominator ports retained. No truncation or text fitting.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quartic_frozen_native_score_v2 import score
from quadratic_product_core_v1 import pairs
STEM='MATCHED_PARTNER_EXACT_INPUT_FOLD_V1'

def evaluate(packed,x):
    ij=pairs(x.shape[1],x.device);result=[]
    for start in range(0,len(x),8):
        xx=x[start:start+8];result.append((xx[:,ij[0]]*xx[:,ij[1]])@packed.T)
    return torch.cat(result)

@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(body_forwards=0,text_sequences=0,cached_endpoints=128,quadratic_forms=3)));return
    signal.alarm(300);tic=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists()
    sd=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
    source=torch.load(P/'MATCHED_PARTNER_SUBSPACE_V1_PROGRAM.pt',weights_only=True)
    l,r=[sd[f'transformer.h.16.mlp.{n}.weight'].double().cuda() for n in ('Left','Right')];c=source['compiled_producer_readers'][:,[0,3,8]].cuda()
    matrices=[];spectra=[]
    for j in range(3):
        raw=l.T@(c[:,j,None]*r);a=(raw+raw.T)/2;matrices.append(a)
        ev=torch.linalg.eigvalsh(a);energy=ev.square();ordered=energy.sort(descending=True).values;cumulative=ordered.cumsum(0)/energy.sum()
        spectra.append(dict(branch=[0,3,8][j],rank90=int(torch.searchsorted(cumulative,torch.tensor(.9,device='cuda')))+1,
                            rank99=int(torch.searchsorted(cumulative,torch.tensor(.99,device='cuda')))+1,positive=int((ev>0).sum()),negative=int((ev<0).sum()),
                            radial_fraction=float(a.trace().square()/1152/a.square().sum()),capture16=float(cumulative[15])))
    matrices=torch.stack(matrices);ij=pairs(1152,'cuda');packed=matrices[:,ij[0],ij[1]]*torch.where(ij[0]==ij[1],1.,2.)[None,:]
    writers=source['output_writers'][:,[3,8]].cuda()
    cache=torch.load(P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_PORTS.pt',weights_only=True);x=cache['ports']['input16'].double().cuda();den=cache['ports']['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    ref=((x@l.T)*(x@r.T))@c;dense=torch.einsum('ni,kij,nj->nk',x,matrices,x);fp64=evaluate(packed,x)
    errors=[float((q-ref).norm()/ref.norm()) for q in (dense,fp64)]
    torch.save(dict(upper_coefficients=packed.float().cpu(),writers=writers.float().cpu(),branches=[3,8],dimension=1152,
                    coefficient_order='torch.triu_indices; off-diagonal coefficients doubled',
                    ports='normalized MLP16 input x16; actual squared MLP17 input RMS denominator',
                    execution='q=upper(x x^T) @ coefficients.T; branch_j=q0*qj*writer_j/den17'),ap)
    saved=torch.load(ap,weights_only=True);round_values=evaluate(saved['upper_coefficients'].double().cuda(),x);round_writers=saved['writers'].double().cuda()
    params=sum(v.numel() for v in saved.values() if torch.is_tensor(v));assert params==1994688
    rows=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_ROWS.json').read_text())['rows'];old=json.loads((P/'MATCHED_PARTNER_FRESH_CONSTRUCTION_V1_RESULT.json').read_text());reports=[]
    torch.set_default_dtype(torch.float32)
    for j,branch in enumerate((3,8)):
        reference=cache['writes'][str(branch)];exact=fp64[:,0,None]*fp64[:,j+1,None]*writers[:,j][None,:]/den[:,None]
        errors.append(float((exact.cpu()-reference).norm()/reference.norm()))
        candidate=round_values[:,0,None]*round_values[:,j+1,None]*round_writers[:,j][None,:]/den[:,None]
        previous=old['reports'][j]
        effects=score([reference,reference,candidate.cpu()],cache['ports']['pre']+cache['ports']['native_output'],rows,sd['lm_head.weight'].float(),reference_effects=dict(swaps=[previous['swaps']],zero_ce=[previous['zero_ce']]))
        reports.append(dict(branch=branch,relative_write_error=float((candidate.cpu()-reference).norm()/reference.norm()),effects=effects))
    result={'pred_a':max(errors)<=1e-8,'pred_b':all(r['relative_write_error']<=1e-5 and r['effects']['pred_a'] and r['effects']['pred_b'] and r['effects']['pred_c'] for r in reports),
            'pred_c':params==1994688 and 10632960*4/ap.stat().st_size>=4}
    result.update(identity_errors=errors,reports=reports,spectra=spectra,conditional_floats=params,single_branch_floats=1329408,
                  native_factor_joint_floats=10632960,artifact_bytes=ap.stat().st_size,byte_reduction=10632960*4/ap.stat().st_size,
                  artifact_sha=digest(ap),source_shas=binding,execution_seconds=time.perf_counter()-tic,
                  scope='Exact algebra with validated FP32 coefficients; compact conditional quartic program, native input/background ports retained, no rank truncation or corpus OOD.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('source_shas','reports')}),flush=True)

if __name__=='__main__':main()
