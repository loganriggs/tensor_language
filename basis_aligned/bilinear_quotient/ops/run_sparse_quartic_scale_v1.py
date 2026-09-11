#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;2x60seconds fixedsupport scale check.
"""pred_a exact gradient rescaling; pred_b unitenergy gradient convergence; pred_c gain<=1e-5.
Original initial-energy-scaled pilot convergence miss remains unchanged.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_quartic_core_v1 import coefficients,indices,fit_fixed
STEM='SPARSE_QUARTIC_SCALE_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    prior=json.loads((P/'SPARSE_QUARTIC_NATIVE_V1_RESULT.json').read_text());source=P/'SPARSE_QUARTIC_NATIVE_V1_PROGRAMS.pt';assert prior['pred_a'] and digest(source)==prior['artifact_sha256']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,max_fit_seconds=120,scale_check=True)));return
    out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(180)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();j=torch.linalg.cholesky(u.T@u).T;del u
    w=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')];w[-1]=j@w[-1]
    programs=torch.load(source,weights_only=True,map_location='cpu');assert programs['complete'];reports=[];errors=[];scale=float(state['transformer.h.17.lambdas'][0])
    allterms,allm=indices(16,'cuda')
    for p in programs['programs']:
        bank=p['bank'].cuda();terms=p['terms'].cuda();m=p['multiplicity_root'].cuda();old=next(r for r in prior['reports'] if r['seed']==p['seed'])
        x=bank.detach().requires_grad_();c=coefficients(w,x,terms,m,scale);energy=c.square().sum()
        g0=torch.autograd.grad(-energy/old['initial_energy'],x,retain_graph=True)[0]
        g1=torch.autograd.grad(-energy/old['selected_energy'],x)[0]
        errors.append(float((g1-g0/old['gain']).norm()/g1.norm()));errors.append(abs(float(energy.detach())/old['selected_energy']-1))
        fitted,r=fit_fixed(w,bank,terms,m,old['selected_energy'],scale,seconds=60,tolerance=1e-7)
        with torch.no_grad():
            c=coefficients(w,fitted,allterms,allm,scale);chosen=c.square().sum(0).topk(128).indices
            expected={tuple(t) for t in terms.T.tolist()};observed={tuple(t) for t in allterms[:,chosen].T.tolist()}
            unchanged=expected==observed;errors.append(0. if unchanged else 1.)
        reports.append(dict(seed=p['seed'],normalized_report=r,energy_relative_change=r['capture']-1,support_unchanged=unchanged))
    result={'pred_a':max(errors)<=1e-8,'pred_b':all(r['normalized_report']['tangent_norm']<=1e-7 and r['normalized_report']['relative_stationarity']<=1e-7 for r in reports),
        'pred_c':all(abs(r['energy_relative_change'])<=1e-5 for r in reports),'maximum_replay_error':max(errors),'reports':reports,
        'scope':'Explicit unitenergy objective convergence diagnostic; original pilot absolute-scale miss preserved. No new artifact or global optimum claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']


if __name__=='__main__':main()
