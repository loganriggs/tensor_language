#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_conditional pred_c_fidelity
"""Full quadratic broad-output pruning baseline; 12 exact writer refits,0forwards.
Instrument relative normal residual<1e-8 and no refit worsening.
Conditional deletion ordering beats energy ordering every budget/geometry;
3686retained channels achieves<10%own-metric coefficient error bothgeometries.
Singleton scores are not additive deletion costs or optimal subset guarantees.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from channel_deletion_cost import costs,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,widths=[2304,3072,3686],forwards=0)));return
 torch.set_grad_enabled(False);start=time.monotonic();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 C0=RU@w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];O=w('transformer.h.17.attn.c_proj.weight');S=torch.linalg.cholesky(torch.eye(1152,device='cuda',dtype=torch.float64)+D@D.T+O@O.T)
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h=data['h'].cuda().double();h=h/(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();h-=h.mean(0);ev,V=torch.linalg.eigh(h.T@h/len(h));cov=(V*ev.clamp_min(1e-8*ev.mean()).sqrt())@V.T;rows=[]
 for geometry,root in [('folded_euclidean',S),('activation_covariance',cov)]:
  A=L@root;B=R@root;an=A.norm(dim=1);bn=B.norm(dim=1);A/=an[:,None];B/=bn[:,None];C=C0*an*bn;C/=C.norm();ab=A@B.T;K=.5*((A@A.T)*(B@B.T)+ab*ab.T);cross=C@K;energy=float((cross*C).sum());score=costs(C,K);orders={'energy':(C.square().sum(0)*K.diag()).argsort(descending=True),'conditional':score.argsort(descending=True)}
  for policy,order in orders.items():
   for width in (2304,3072,3686):
    ids=order[:width];ks=K[ids][:,ids];rhs=cross[:,ids];writer=torch.cholesky_solve(rhs.T,torch.linalg.cholesky(ks)).T;residual=float((writer@ks-rhs).norm()/rhs.norm());error=max(0,1-float((writer*rhs).sum())/energy)**.5;before=max(0,(energy+float(((C[:,ids]@ks)*C[:,ids]).sum())-2*float((C[:,ids]*rhs).sum()))/energy)**.5
    row=dict(geometry=geometry,policy=policy,width=width,relative_error=error,unrefitted_error=before,normal_residual=residual,reduced_factor_coefficients=3*1152*width,variable_products=width,saving_fraction=1-width/4608,selected_channels=ids.cpu().tolist());rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='selected_channels'}),flush=True)
 pred=dict(pred_a_instrument=all(r['normal_residual']<1e-8 and r['relative_error']<=r['unrefitted_error']+1e-8 for r in rows),pred_b_conditional=all(r['relative_error']<=next(t['relative_error'] for t in rows if t['geometry']==r['geometry'] and t['width']==r['width'] and t['policy']=='energy') for r in rows if r['policy']=='conditional'),pred_c_fidelity=all(min(r['relative_error'] for r in rows if r['geometry']==g and r['width']==3686)<.1 for g in ('folded_euclidean','activation_covariance')))
 (P/'FULL_CHANNEL_DELETION_V1.json').write_text(json.dumps(dict(records=rows,predictions=pred,controls=check,seconds=time.monotonic()-start,scope='Fullquadratic numerator broad-output teacher-channel pruning and exact writer refit. Samebasisgeometry asmodebounds. Singleton conditionalimportance is not exactbulkdeletion or bestsubset. Framecost shared/excluded, no learnedcircuitidentity orbehavioralvalidation.'),indent=2)+'\n')
if __name__=='__main__':main()
