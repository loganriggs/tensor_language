#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;128cachedvectors,64inner+4outer eigensolves, nofit.
"""pred_a nativefold/reader/executable identities<=1e-8; pred_b outer16error<=.1; pred_c compacterror<=.1.
Fixed16x16 hierarchy per2outputmodes perseed,592672floats each; weights-only eigensolves.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from fullsource_quartic_program_v1 import run
STEM='FULLSOURCE_QUARTIC_SPECTRAL_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def relative(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    pr=json.loads((P/'QUARTIC_GROUP_BOUNDARY_V1.json').read_text());targetfile=P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt';assert pr['pred_a'] and digest(targetfile)==pr['artifact_sha256']
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,cached_vectors=128,eigensolves=68,fitting=False,fitted_floats=592672)));return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(180)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
    l0,r0,d0,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    scale=float(state['transformer.h.17.lambdas'][0]);x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True,map_location='cpu')['input16'].double().cuda()
    old=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True,map_location='cpu')['ports'];den=old['pre'].double().cuda().square().mean(-1)+torch.finfo(torch.float32).eps
    producer=((x@l0.T)*(x@r0.T))@d0.T*scale
    groups=torch.load(P/'QUARTIC_GROUP_PROGRAM_V1.pt',weights_only=True,map_location='cpu')['programs'];refs=torch.load(targetfile,weights_only=True,map_location='cpu')['lifted'];reports=[];saved=[];errors=[]
    for idx,g in enumerate(groups):
        writer=g['output_writers'].cuda();readers=metric@writer;inner_b=[];inner_w=[];outer_w=[];outer_scalar=[];exact_scalar=[];spectra=[]
        for mode in range(2):
            coefficient=d1.T@readers[:,mode];a=l1.T@(coefficient[:,None]*r1);matrix=(a+a.T)/2
            scalar=(producer*(producer@matrix)).sum(-1);exact_scalar.append(scalar)
            eig,vec=torch.linalg.eigh(matrix);chosen=eig.abs().topk(16).indices;mu=eig[chosen];outer=vec[:,chosen];outer_w.append(mu)
            outer_scalar.append(((producer@outer).square()*mu).sum(-1));bs=[];nus=[];inner_capture=[]
            for j in range(16):
                coefficient=d0.T@outer[:,j]*scale;a=l0.T@(coefficient[:,None]*r0);matrix=(a+a.T)/2
                direct=(x*(x@matrix)).sum(-1);native=producer@outer[:,j];errors.append(relative(direct,native))
                val,axis=torch.linalg.eigh(matrix);selected=val.abs().topk(16).indices
                bs.append(axis[:,selected]);nus.append(val[selected]);inner_capture.append(float(val[selected].square().sum()/val.square().sum()))
            inner_b.append(torch.stack(bs));inner_w.append(torch.stack(nus));spectra.append(dict(mode=mode,outer_quadratic_energy_fraction=float(mu.square().sum()/eig.square().sum()),inner_quadratic_energy_fractions=inner_capture))
        exact=torch.stack(exact_scalar,1)@writer.T/den[:,None];ref=refs[idx].cuda();errors.append(relative(exact,ref))
        outerwrite=torch.stack(outer_scalar,1)@writer.T/den[:,None]
        program=dict(seed=g['seed'],input_readers=torch.stack(inner_b),inner_weights=torch.stack(inner_w),outer_weights=torch.stack(outer_w),output_writers=writer)
        assert sum(v.numel() for v in program.values() if torch.is_tensor(v))==592672
        compact=run(program,x,den);assert torch.isfinite(compact).all()
        explicit=[]
        for mode in range(2):
            terms=[]
            for j in range(16):
                q=((x@inner_b[mode][j]).square()*inner_w[mode][j]).sum(-1);terms.append(outer_w[mode][j]*q.square())
            explicit.append(torch.stack(terms).sum(0))
        errors.append(relative(compact,torch.stack(explicit,1)@writer.T/den[:,None]))
        reports.append(dict(seed=g['seed'],outer16_relative_error=relative(outerwrite,ref),compact_relative_error=relative(compact,ref),spectra=spectra))
        saved.append({k:v.cpu() if torch.is_tensor(v) else v for k,v in program.items()});print(json.dumps(reports[-1]),flush=True)
    torch.save(dict(programs=saved),ap)
    result={'pred_a':max(errors)<=1e-8,'pred_b':all(r['outer16_relative_error']<=.1 for r in reports),'pred_c':all(r['compact_relative_error']<=.1 for r in reports),
        'maximum_identity_error':max(errors),'reports':reports,'artifact_sha256':digest(ap),'artifact_bytes':ap.stat().st_size,'wall_seconds':time.perf_counter()-started,
        'scope':'Fixedrank weight-only spectral hierarchy with separateinputspaces, developmental cachedwrite validation. Notquartic-optimal or behavioral/OODcertified.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2),flush=True);assert result['pred_a']


if __name__=='__main__':main()
