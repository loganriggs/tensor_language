#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;2frozen quartic banks, no fit,180secondalarm.
"""pred_a replay/energy identities; pred_b saved128>=.8dense; pred_c outside>=10discarded.
Terminal-only producer dependency, exact fixed-bank coefficients, full/centered output spectra.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_quartic_core_v1 import coefficients,indices
STEM='SPARSE_QUARTIC_CEILING_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,fitting=False,requires_completed_producer=True)));return
    priorfile=P/'SPARSE_QUARTIC_NATIVE_V1_RESULT.json';prior=json.loads(priorfile.read_text());assert prior['pred_a']
    source=P/'SPARSE_QUARTIC_NATIVE_V1_PROGRAMS.pt';assert digest(source)==prior['artifact_sha256']
    artifact=torch.load(source,weights_only=True,map_location='cpu');assert artifact['complete'] and len(artifact['programs'])==2
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_MODES.pt');assert not out.exists() and not ap.exists();signal.alarm(180)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();g=u.T@u;mu=u.mean(0);gc=g-len(u)*torch.outer(mu,mu);del u
    root=torch.linalg.cholesky(g).T;center=torch.linalg.cholesky(gc).T
    transform=torch.linalg.solve_triangular(root.T,center.T,upper=False).T
    weights=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    weights[-1]=root@weights[-1];scale=float(state['transformer.h.17.lambdas'][0]);terms,m=indices(16,'cuda')
    lookup={tuple(t):i for i,t in enumerate(terms.T.tolist())};errors=[];ortherrors=[];reports=[];modes=[]
    for program in artifact['programs']:
        bank=program['bank'].cuda();c=coefficients(weights,bank,terms,m,scale)
        selected=torch.tensor([lookup[tuple(t)] for t in program['terms'].T.tolist()],device='cuda')
        expected=root@program['physical_writer'].cuda();errors.append(float((c[:,selected]-expected).norm()/expected.norm()))
        ortherrors.append(float((bank.T@bank-torch.eye(16,device='cuda')).abs().max()))
        energy=c.square().sum(0);saved=float(energy[selected].sum());dense=float(energy.sum());best=float(energy.topk(128).values.sum())
        old=next(r for r in prior['reports'] if r['seed']==program['seed']);errors.append(abs(saved/old['selected_energy']-1))
        assert saved<=best*(1+1e-10) and best<=dense*(1+1e-10)
        spectra=[]
        for name,j,cview in [('full',root,c),('centered',center,transform@c)]:
            eig,vec=torch.linalg.eigh(cview@cview.T);order=torch.arange(len(eig)-1,-1,-1,device='cuda');eig=eig[order];vec=vec[:,order]
            total=float(cview.square().sum());errors.append(abs(float(eig.sum())/total-1));assert eig.min()>=-1e-10*eig.max()
            direction=vec[:,:8];scalars=direction.T@cview;physical=torch.linalg.solve_triangular(j,direction,upper=True)
            errors.append(float(((j@physical).T@(j@physical)-torch.eye(8,device='cuda')).abs().max()))
            modes.append(dict(seed=program['seed'],metric=name,bank=bank.cpu(),terms=terms.cpu(),multiplicity_root=m.cpu(),physical_writer=physical.cpu(),scalar_coefficients=scalars.cpu()))
            spectra.append(dict(metric=name,projected_energy=total,leading_energy_fractions={str(k):float(eig[:k].sum()/total) for k in [1,2,4,8,16,32,64]},
                leading16_eigenvalues=eig[:16].tolist()))
        discarded=dense-saved;outside=prior['reference_norm2']-dense
        reports.append(dict(seed=program['seed'],saved_energy=saved,best128_energy=best,dense_projected_energy=dense,
            saved_fraction_of_dense=saved/dense,estimated_dense_native_capture=dense/prior['reference_norm2'],
            discarded_inside_energy=discarded,estimated_outside_energy=outside,estimated_outside_over_discarded=outside/max(discarded,1e-30),spectra=spectra))
    torch.save(dict(modes=modes),ap)
    result={'pred_a':max(errors)<=1e-8 and max(ortherrors)<=1e-9,'pred_b':all(r['saved_fraction_of_dense']>=.8 for r in reports),
        'pred_c':all(r['estimated_outside_over_discarded']>=10 for r in reports),'reports':reports,'maximum_replay_error':max(errors),'orthogonality_errors':ortherrors,
        'source_result_sha256':digest(priorfile),'source_artifact_sha256':digest(source),'artifact_sha256':digest(ap),'artifact_bytes':ap.stat().st_size,
        'wall_seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Exact fixed-bank ceiling and descriptive output modes; fullnorm estimated, no optimization or global rank/circuit claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']


if __name__=='__main__':main()
