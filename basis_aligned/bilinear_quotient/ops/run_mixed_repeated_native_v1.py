#!/usr/bin/env python3
# BQGATE:0bodyforwards;4grades2048syntheticprobes;batch64;240sec.
"""pred_a graded diagonal/oracle<=1e-8/grade1within.02+3SE;
pred_b mixedgradegap>.05+2SE; pred_c relativeSE<=.1/runtime<=180sec.
Null: no resolved mixed repeated-input metric difference.
"""
import sys,os,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from mixed_repeated_contraction_v1 import contract
from composed_eighth_contraction_v2 import contract as eighth
from attention_source_quartic_v1 import compile_sources
from graded_source_projection_v1 import graded_norms,retained_grades
STEM='MIXED_REPEATED_NATIVE_V1'
@torch.no_grad()
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('4 grades x 2048 synthetic probes; 0 body forwards');return
    out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_ENERGIES.pt');assert not out.exists() and not ap.exists();signal.alarm(240);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
    def weight(k):return state[k].double().cuda()
    l=weight('transformer.h.15.mlp.Left.weight');r=weight('transformer.h.15.mlp.Right.weight');d=weight('transformer.h.15.mlp.Down.weight')*weight('transformer.h.16.lambdas')[0]
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();wg=data['writer_gram'].cuda();root=data['root'].cuda();root=root/root.square().sum().div(len(root)).sqrt()
    program=torch.load(P/'GRADED_SOURCE_LBFGS_V1_PROGRAM.pt',weights_only=True,map_location='cpu');downs=[d]+[z['write'].double().cuda()@(z['read'].double().cuda()@d) for z in program['programs']]
    full=graded_norms(a,root@root.T,wg);formal=[(1-retained_grades(a,root,p.cuda(),wg)/full).clamp_min(0).sqrt().cpu() for p in program['frames']]
    gen=torch.Generator(device='cuda').manual_seed(73210);b=torch.randn(8,1152,device='cuda',dtype=torch.float64,generator=gen);x=torch.randn(b.shape,device=b.device,dtype=b.dtype,generator=gen);m=((x@l.T)*(x@r.T))@d.T
    packed=torch.load(P/'MATCHED_PARTNER_EXACT_INPUT_FOLD_V1_PROGRAM.pt',weights_only=True);reference=compile_sources(packed,b,m)['sectors'];checks=[]
    for grade in range(5):
        predicted=contract(b[:,None,:].expand(-1,4-grade,-1),x[:,None,:].expand(-1,2*grade,-1),l,r,d,a,grade)
        checks.append(float((predicted-reference[:,:,grade]).norm()/reference[:,:,grade].norm()))
    x=torch.randn(4,8,1152,device='cuda',dtype=torch.float64,generator=gen);q=eighth(x,l,r,d,a);z=contract(x[:,:0],x,l,r,d,a,4);checks.append(float((z-q).norm()/q.norm()))
    energies={};reports=[]
    for grade in range(1,5):
        values=[]
        for start in range(0,2048,64):
            b=torch.randint(0,2,(64,4-grade,1152),device='cuda',generator=gen).double()*2-1;x=torch.randint(0,2,(64,2*grade,1152),device='cuda',generator=gen).double()*2-1
            ref=contract(b,x,l,r,d,a,grade);row=[torch.einsum('ni,ij,nj->n',ref,wg,ref)]
            for dd in downs[1:]:
                error=contract(b,x,l,r,dd,a,grade)-ref;row.append(torch.einsum('ni,ij,nj->n',error,wg,error))
            values.append(torch.stack(row,-1).cpu())
        e=torch.cat(values);assert len(e)==2048 and bool(torch.isfinite(e).all());energies[str(grade)]=e;ref=e[:,0]
        for arm in range(2):
            error=e[:,arm+1];ratio=error.mean()/ref.mean();estimate=ratio.sqrt();se=(error-ratio*ref).std(unbiased=True)/(len(ref)**.5*ref.mean()*2*estimate)
            reports.append(dict(grade=grade,arm=arm,relative_error=float(estimate),standard_error=float(se),relative_standard_error=float(se/estimate),formal_relative_error=float(formal[arm][grade]),gap=float(estimate-formal[arm][grade])))
    seconds=time.perf_counter()-tic
    result={'pred_a':max(checks)<=1e-8 and all(abs(z['gap'])<=.02+3*z['standard_error'] for z in reports if z['grade']==1),
            'pred_b':any(abs(z['gap'])>.05+2*z['standard_error'] for z in reports if z['grade'] in (2,3)),
            'pred_c':all(z['relative_standard_error']<=.1 for z in reports) and seconds<=180}
    torch.save(dict(energies=energies,seed=73210),ap);result.update(checks=checks,reports=reports,artifact_sha=digest(ap),source_shas=binding,execution_seconds=seconds)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
