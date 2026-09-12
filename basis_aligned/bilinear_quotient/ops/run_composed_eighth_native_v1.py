#!/usr/bin/env python3
# BQGATE:0bodyforwards;2048synthetic8slotprobes;batch64;180sec.
"""pred_a native diagonal/permutation<=1e-8; pred_b metricgap>.05+2SE;
pred_c relativeSE<=.1/runtime<=120sec. Null: no resolved metric difference.
"""
import sys,os,json,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from composed_eighth_contraction_v2 import contract
from graded_source_projection_v1 import graded_norms,retained_grades
STEM='COMPOSED_EIGHTH_NATIVE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('2048 independent eight-slot probes; 0 body forwards');return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_ENERGIES.pt');assert not out.exists() and not ap.exists();signal.alarm(180);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(k):return state[k].double().cuda()
    l=weight('transformer.h.15.mlp.Left.weight');r=weight('transformer.h.15.mlp.Right.weight');d=weight('transformer.h.15.mlp.Down.weight')*weight('transformer.h.16.lambdas')[0]
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();wg=data['writer_gram'].cuda();root=data['root'].cuda();root=root/root.square().sum().div(len(root)).sqrt()
    program=torch.load(P/'GRADED_SOURCE_LBFGS_V1_PROGRAM.pt',weights_only=True,map_location='cpu');downs=[d]+[z['write'].double().cuda()@(z['read'].double().cuda()@d) for z in program['programs']]
    full=graded_norms(a,root@root.T,wg);formal=[float((1-retained_grades(a,root,p.cuda(),wg)[4]/full[4]).clamp_min(0).sqrt()) for p in program['frames']]
    gen=torch.Generator(device='cuda').manual_seed(73190);x=torch.randn(8,1152,device='cuda',dtype=torch.float64,generator=gen);m=((x@l.T)*(x@r.T))@d.T;q=torch.einsum('ni,kij,nj->nk',m,a,m);direct=q[:,0,None]*q[:,1:];diagonal=contract(x[:,None,:].expand(-1,8,-1),l,r,d,a)
    checks=[float((diagonal-direct).norm()/direct.norm())];probe=torch.randn(4,8,1152,device='cuda',dtype=torch.float64,generator=gen);ref=contract(probe,l,r,d,a);perm=contract(probe[:,[7,2,0,5,3,1,6,4]],l,r,d,a);checks.append(float((perm-ref).norm()/ref.norm()))
    values=[]
    for start in range(0,2048,64):
        x=torch.randint(0,2,(64,8,1152),device='cuda',generator=gen).double()*2-1
        ref=contract(x,l,r,d,a);row=[torch.einsum('ni,ij,nj->n',ref,wg,ref)]
        for dd in downs[1:]:
            error=contract(x,l,r,dd,a)-ref;row.append(torch.einsum('ni,ij,nj->n',error,wg,error))
        values.append(torch.stack(row,-1).cpu())
    energies=torch.cat(values);reports=[];reference=energies[:,0]
    for arm in range(2):
        error=energies[:,arm+1];ratio=error.mean()/reference.mean();estimate=ratio.sqrt();se_ratio=(error-ratio*reference).std(unbiased=True)/(len(error)**.5*reference.mean());se=se_ratio/(2*estimate)
        reports.append(dict(arm=arm,eighth_relative_error=float(estimate),standard_error=float(se),relative_standard_error=float(se/estimate),formal_relative_error=formal[arm],gap=float(estimate)-formal[arm]))
    seconds=time.perf_counter()-tic
    result={'pred_a':max(checks)<=1e-8 and len(energies)==2048 and bool(torch.isfinite(energies).all()) and float(reference.mean())>0,
            'pred_b':any(abs(z['gap'])>.05+2*z['standard_error'] for z in reports),'pred_c':all(z['relative_standard_error']<=.1 for z in reports) and seconds<=120}
    torch.save(dict(energies=energies,seed=73190),ap)
    result.update(checks=checks,reports=reports,reference_mean=float(reference.mean()),artifact_sha=digest(ap),source_shas=binding,execution_seconds=seconds,scope='Homogeneous pure-producer coefficient metric only; mixed/native normalized circuit remains untested here.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
