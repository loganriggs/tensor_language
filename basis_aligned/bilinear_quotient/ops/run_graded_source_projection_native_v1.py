#!/usr/bin/env python3
# BQGATE:0bodyforwards;cachednativecoefficientobjective;120sec.
"""pred_a grade sum<=1e-8/tangentFD<=1e-5; pred_b step improves>=1e-6;
pred_c gradient<=2sec. Null: graded weight-only metric unusable.
"""
import sys,os,time,json,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from graded_source_projection_v1 import graded_norms,retained_grades,balanced_loss
from coupled_source_projection_v1 import tangent,retract
STEM='GRADED_SOURCE_PROJECTION_NATIVE_V1'
def main():
    binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0 body forwards; cached weights; 120 seconds');return
    out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    data=torch.load(P/'COUPLED_SOURCE_PROJECTION_NATIVE_V1_INPUTS.pt',weights_only=True,map_location='cpu');a=data['forms'].cuda();rawroot=data['root'].cuda();wg=data['writer_gram'].cuda()
    raw=graded_norms(a,rawroot@rawroot.T,wg);scale=rawroot.square().sum().div(len(rawroot)).sqrt();root=rawroot/scale;full=graded_norms(a,root@root.T,wg)
    p=data['starts']['128'].cuda().requires_grad_();torch.cuda.synchronize();tic=time.perf_counter();loss=balanced_loss(a,root,p,wg,full);grad=tangent(p,torch.autograd.grad(loss,p)[0]);torch.cuda.synchronize();seconds=time.perf_counter()-tic
    with torch.no_grad():
        direction=-grad/grad.norm();eps=1e-4
        def value(q):return balanced_loss(a,root,q,wg,full)
        fd=(value(retract(p,eps*direction))-value(retract(p,-eps*direction)))/(2*eps);exact=(grad*direction).sum();fd_error=float((fd-exact).abs()/exact.abs())
        step=.1
        for _ in range(12):
            new=float(value(retract(p,step*direction)))
            if new<float(loss):break
            step/=2
        reports={}
        for rank in ('128','512'):
            retained=retained_grades(a,root,data['starts'][rank].cuda(),wg)/full
            reports[rank]=dict(retention=retained.tolist(),loss=float((1-retained[1:]).mean()))
    sum_error=abs(float(raw.sum())-data['full'])/data['full']
    result={'pred_a':bool((full>0).all()) and sum_error<=1e-8 and fd_error<=1e-5,'pred_b':float(loss)-new>=1e-6,'pred_c':seconds<=2}
    result.update(raw_grade_fractions=(raw/raw.sum()).tolist(),root_scale=float(scale),reports=reports,gradient_seconds=seconds,tangent_norm=float(grad.norm()),finite_difference_error=fd_error,sum_error=sum_error,one_step_loss=new,improvement=float(loss)-new,source_shas=binding)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}))
if __name__=='__main__':main()
