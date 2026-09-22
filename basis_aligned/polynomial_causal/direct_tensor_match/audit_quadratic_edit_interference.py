"""Separate joint-edit interference from Gaussian-to-text metric transfer."""
import json,time
import torch
from correlated_gaussian_cp import gram
from audit_conditional_residual_accounting import P,SCALE,load

def energy(K,C):
 return (K*(C.T@C)).sum()

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 saved=json.loads((P/'CP_QUADRATIC_REUSE_V1.json').read_text())
 cache=torch.load(P/'GAUSSIAN_CP_DATA_PROJECTIONS_V1.pt',weights_only=True)
 S=cache['projections']['covariance']['whitener'].double();mu=cache['mean'].double()
 data=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True);x=data['rows'].double()
 rows=[]
 for proposal in saved['proposals']:
  seed=proposal['seed'];p,sha=load(f'MIXED_CP_FEATURES_SEED{seed}_V1.pt');assert sha==proposal['parent_sha256']
  fs=p['factors'];C=p['coefficients']/SCALE
  F=[f@S for f in fs];b=[f@mu for f in fs]
  parent_gauss=energy(gram(F,b,F,b),C)
  values=torch.stack([x@f.T for f in fs]).prod(0)
  parent_text=(values@C.T).square().sum()/len(x)
  for count in [64,128]:
   edits=proposal['selected'][:count];atoms=torch.tensor([i%512 for i,j,scale in edits]);scales=torch.tensor([scale for i,j,scale in edits],dtype=torch.float64)
   old=[f[atoms] for f in fs];new=[f.clone() for f in old]
   for k,(i,j,scale) in enumerate(edits):
    for slot in [0,1]:new[2*(i//512)+slot][k]=fs[2*(j//512)+slot][j%512]
   of=[f@S for f in old];ob=[f@mu for f in old];nf=[f@S for f in new];nb=[f@mu for f in new]
   oo=gram(of,ob,of,ob);nn=gram(nf,nb,nf,nb);on=gram(of,ob,nf,nb)
   K=oo+scales[:,None]*nn*scales[None,:]-on*scales[None,:]-on.T*scales[:,None]
   c=C[:,atoms];gn=energy(K,c);gd=(K.diag()*c.square().sum(0)).sum()
   delta=torch.stack([x@f.T for f in new]).prod(0)*scales-values[:,atoms]
   empirical=delta.T@delta/len(x);tn=energy(empirical,c);td=(empirical.diag()*c.square().sum(0)).sum()
   ge=float((gn/parent_gauss).clamp_min(0).sqrt());te=float((tn/parent_text).sqrt())
   reference=next(r for r in saved['rows'] if r['seed']==seed and r['count']==count and r['panel']=='opened256')
   replay=abs(te-reference['parent_value_error']);assert replay<1e-10
   row=dict(seed=seed,count=count,gaussian_joint_over_sum_individual=float(gn/gd),text_joint_over_sum_individual=float(tn/td),gaussian_relative_edit=ge,text_relative_edit=te,text_over_gaussian_relative_edit=te/ge,replay=replay)
   rows.append(row);print(row,flush=True)
 out=dict(rows=rows,seconds=time.monotonic()-start,scope='Descriptive diagnosis of already opened candidates. Ratios of joint to individual squared edit energies quantify interference. Text/Gaussian compares normalized RMS edit, not the same input distribution. No refit or selection.')
 (P/'QUADRATIC_EDIT_INTERFERENCE_V1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
