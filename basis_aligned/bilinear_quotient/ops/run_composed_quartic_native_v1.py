#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;8192synthetic quartic coefficient probes.
"""pred_a exact native contraction controls, pred_b estimatedrelativeSE<=.05 bothdistributions,
pred_c Gaussian/Rademacher estimates agree within3combinedestimatedSE.
Pure degree4 producer/producer route. Bias/RMS/otherpaths external, no sparsefit.
"""
import os,sys,time,json,hashlib,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from composed_quartic_contraction_v1 import contract
STEM='COMPOSED_QUARTIC_NATIVE_V1'


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def pair_partition(x,w,scale):
    l0,r0,d0,l1,r1,d1=w;l=x@l0.T;r=x@r0.T
    first=((l[:,0]*r[:,1]+r[:,0]*l[:,1])/2)@d0.T*scale
    second=((l[:,2]*r[:,3]+r[:,2]*l[:,3])/2)@d0.T*scale
    return ((first@l1.T)*(second@r1.T)+(first@r1.T)*(second@l1.T))@d1.T/2


@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    control=json.loads((P/'COMPOSED_QUARTIC_CONTRACTION_V1_CONTROL.json').read_text());assert all(v for k,v in control.items() if k.startswith('pred_'))
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,synthetic_probes=8192,batch_size=64,fitting=False)));return
    out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(300)
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;started=time.perf_counter()
    ck=next(k for k in binding if k.endswith('/pytorch_model.bin'));state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
    u=state['lm_head.weight'].double().cuda();root=torch.linalg.cholesky(u.T@u).T;del u
    w=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
    w[-1]=root@w[-1];scale=float(state['transformer.h.17.lambdas'][0]);torch.manual_seed(11410)
    x=torch.randn(8,4,1152,device='cuda');value=contract(x,*w,scale);permuted=contract(x[:,[2,0,3,1]],*w,scale)
    permerror=float((value-permuted).norm()/value.norm())
    v=x[:,0];l0,r0,d0,l1,r1,d1=w;p=((v@l0.T)*(v@r0.T))@d0.T*scale;native=((p@l1.T)*(p@r1.T))@d1.T
    diagonal=contract(v[:,None,:].expand(-1,4,-1),*w,scale);diagerror=float((diagonal-native).norm()/native.norm());assert max(permerror,diagerror)<=1e-10
    results=[]
    for distribution,seed in [('gaussian',11411),('rademacher',11412)]:
        torch.manual_seed(seed);samples=[];pairs=[];torch.cuda.synchronize();tic=time.perf_counter()
        for batch in range(64):
            x=torch.randn(64,4,1152,device='cuda') if distribution=='gaussian' else (torch.randint(0,2,(64,4,1152),device='cuda').double()*2-1)
            samples.append(contract(x,*w,scale).square().sum(-1));pairs.append(pair_partition(x,w,scale).square().sum(-1))
        torch.cuda.synchronize();seconds=time.perf_counter()-tic
        values=torch.cat(samples);pairvalues=torch.cat(pairs);assert len(values)==4096 and torch.isfinite(values).all() and values.mean()>0
        mean=float(values.mean());se=float(values.std(unbiased=True)/len(values)**.5)
        results.append(dict(distribution=distribution,seed=seed,probes=len(values),estimated_norm2=mean,estimated_standard_error=se,relative_standard_error=se/mean,
            unsymmetrized_pair_partition_norm2=float(pairvalues.mean()),symmetric_to_pair_norm2_ratio=float(values.mean()/pairvalues.mean()),seconds=seconds))
        print(json.dumps(results[-1]),flush=True)
    discrepancy=abs(results[0]['estimated_norm2']-results[1]['estimated_norm2'])
    combined=(results[0]['estimated_standard_error']**2+results[1]['estimated_standard_error']**2)**.5
    result={'pred_a':max(permerror,diagerror)<=1e-10,'pred_b':all(r['relative_standard_error']<=.05 for r in results),
        'pred_c':discrepancy<=3*combined,'permutation_error':permerror,'diagonal_error':diagerror,'discrepancy_in_estimated_standard_errors':discrepancy/combined,
        'results':results,'lambda17_0':scale,'wall_seconds':time.perf_counter()-started,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Native full-U homogeneous quartic coefficient oracle and sampling cost. Estimated SE not rigorous error bound. No text, sparse fit, full normalized model equivalence or circuit claim.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']

if __name__=='__main__':main()
