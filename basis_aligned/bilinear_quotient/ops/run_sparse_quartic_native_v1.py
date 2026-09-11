#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;2x1200softseconds exact weight-only quartic optimization.
"""pred_a native controls; pred_b bothsupport/gradientconverged; pred_c bothgain>=1.10.
Full-U degree4 composition, learned16readers/128quarticedges; no text fitting.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_quartic_core_v1 import coefficients,indices,fit_fixed
from composed_quartic_contraction_v1 import contract
STEM='SPARSE_QUARTIC_NATIVE_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'SPARSE_QUARTIC_CORE_V2_CONTROL.json').read_text());assert control['pred_a'] and control['pred_b']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,max_fit_seconds=2400,readers=16,quartic_edges=128,fitted_floats=165888)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(3000)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();root=torch.linalg.cholesky(u.T@u).T;del u
    weights=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    weights[-1]=root@weights[-1];scale=float(state['transformer.h.17.lambdas'][0]);terms,m=indices(16,'cuda')
    l,r,d=weights[:3];dn=d.square().sum(0);ln=l.square().sum(1);rn=r.square().sum(1)
    proxy=l.T@((dn*rn)[:,None]*l)+r.T@((dn*ln)[:,None]*r)
    spectral=torch.linalg.eigh(proxy).eigenvectors[:,-32:];del proxy
    reference=json.loads((P/'COMPOSED_QUARTIC_NATIVE_V1_RESULT.json').read_text())['results'][0]
    norm2=reference['estimated_norm2'];reports=[];programs=[];errors=[];ortherrors=[];declines=[]
    for seed in [11511,11512]:
        torch.manual_seed(seed);bank=spectral@torch.linalg.qr(torch.randn(32,16,device='cuda')).Q
        with torch.no_grad():
            selected=torch.randperm(terms.shape[1],device='cuda')[:16]
            exact=coefficients(weights,bank,terms[:,selected],m[selected],scale)
            oracle=contract(bank.T[terms[:,selected].T],*weights,scale).T*m[selected]
            errors.append(float((exact-oracle).norm()/oracle.norm()));assert errors[-1]<=1e-8
            tic=time.perf_counter();allc=coefficients(weights,bank,terms,m,scale);torch.cuda.synchronize();allseconds=time.perf_counter()-tic
            support=allc.square().sum(0).topk(128).indices;initial=float(allc[:,support].square().sum());del allc
        previous=initial;history=[];stable=0
        for cycle in range(20):
            with torch.no_grad():support=coefficients(weights,bank,terms,m,scale).square().sum(0).topk(128).indices
            bank,report=fit_fixed(weights,bank,terms[:,support],m[support],initial,scale,seconds=60,tolerance=1e-7)
            with torch.no_grad():
                allc=coefficients(weights,bank,terms,m,scale);energies=allc.square().sum(0);new=energies.topk(128).indices
                energy=float(energies[support].sum());best=float(energies[new].sum());del allc
            same=bool(torch.equal(support.sort().values,new.sort().values));stable=stable+1 if same else 0
            declines.append(max(0.,(previous-energy)/max(previous,1e-30)));previous=energy
            report.update(cycle=cycle,support_unchanged=same,consecutive_stable=stable,selected_energy=energy,estimated_native_capture=energy/norm2,
                best_support_gain=(best-energy)/max(energy,1e-30));history.append(report)
            print(json.dumps(dict(seed=seed,**{k:v for k,v in report.items() if k!='history'})),flush=True)
            (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(dict(completed=reports,current=dict(seed=seed,cycles=history)),indent=2)+'\n')
            if stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-5:break
        with torch.no_grad():
            c=coefficients(weights,bank,terms[:,support],m[support],scale);physical=torch.linalg.solve_triangular(root,c,upper=True)
            orth=float((bank.T@bank-torch.eye(16,device='cuda')).abs().max());ortherrors.append(orth)
            assert torch.isfinite(bank).all() and torch.isfinite(physical).all()
            programs.append(dict(seed=seed,bank=bank.cpu(),terms=terms[:,support].cpu(),multiplicity_root=m[support].cpu(),physical_writer=physical.cpu(),lambda17_0=scale))
            torch.save(dict(programs=programs,complete=len(programs)==2),ap)
        reports.append(dict(seed=seed,initial_energy=initial,selected_energy=energy,estimated_native_capture=energy/norm2,gain=energy/initial,
            converged=stable>=2 and report['tangent_norm']<=1e-7 and report['relative_stationarity']<=1e-5,all_coefficient_seconds=allseconds,
            orthogonality_error=orth,cycles=history))
    result={'pred_a':max(errors)<=1e-8 and max(ortherrors)<=1e-9 and max(declines)<=1e-8,
        'pred_b':all(r['converged'] for r in reports),'pred_c':all(r['gain']>=1.10 for r in reports),'reports':reports,
        'oracle_errors':errors,'maximum_relative_energy_decline':max(declines),'reference_norm2':norm2,
        'reference_relative_standard_error':reference['relative_standard_error'],'artifact_sha256':digest(ap),'artifact_bytes':ap.stat().st_size,
        'wall_seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Exact restricted sparsequartic coefficient optimization, estimated fulltargetnorm only for reported fraction. Localpilot, not circuit/OOD/normalizedmodel equivalence.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2),flush=True);assert result['pred_a']


if __name__=='__main__':main()
