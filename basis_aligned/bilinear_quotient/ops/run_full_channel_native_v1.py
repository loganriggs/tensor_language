#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_endpoint pred_c_repair
"""Frozen full-layer candidates through actual output nonlinearities.
Instrument native residual replay<1e-5; all corrected endpoint errors<10%;
affine correction improves every panel/domain. Opened panels,96captures."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from channel_deletion_cost import costs,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,widths=[3686],captures=96)));return
 torch.set_grad_enabled(False);start=time.monotonic();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 C0=RU@w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];O=w('transformer.h.17.attn.c_proj.weight');S=torch.linalg.cholesky(torch.eye(1152,device='cuda',dtype=torch.float64)+D@D.T+O@O.T)
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h=data['h'].cuda().double();h=h/(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();mu=h.mean(0);x=h.clone();h-=mu;ev,V=torch.linalg.eigh(h.T@h/len(h));cov=(V*ev.clamp_min(1e-8*ev.mean()).sqrt())@V.T;rows=[];programs={}
 for geometry,root in [('folded_euclidean',S),('activation_covariance',cov)]:
  A=L@root;B=R@root;an=A.norm(dim=1);bn=B.norm(dim=1);A/=an[:,None];B/=bn[:,None];C=C0*an*bn;cscale=C.norm();C/=cscale;ab=A@B.T;K=.5*((A@A.T)*(B@B.T)+ab*ab.T);cross=C@K;energy=float((cross*C).sum());score=costs(C,K);orders={'energy':(C.square().sum(0)*K.diag()).argsort(descending=True),'conditional':score.argsort(descending=True)}
  for policy,order in orders.items():
   if policy!=('energy' if geometry=='folded_euclidean' else 'conditional'):continue
   for width in (3686,):
    ids=order[:width];ks=K[ids][:,ids];rhs=cross[:,ids];writer=torch.cholesky_solve(rhs.T,torch.linalg.cholesky(ks)).T;residual=float((writer@ks-rhs).norm()/rhs.norm());error=max(0,1-float((writer*rhs).sum())/energy)**.5;before=max(0,(energy+float(((C[:,ids]@ks)*C[:,ids]).sum())-2*float((C[:,ids]*rhs).sum()))/energy)**.5
    ar=L[ids]/an[ids,None];br=R[ids]/bn[ids,None];teacher=((x@L.T)*(x@R.T))@(C0/cscale).T;student=((x@ar.T)*(x@br.T))@writer.T
    true_at_mu=((mu@L.T)*(mu@R.T))@(C0/cscale).T;student_at_mu=((mu@ar.T)*(mu@br.T))@writer.T;linear=(C0/cscale)@((mu@R.T)[:,None]*L+(mu@L.T)[:,None]*R)-writer@((mu@br.T)[:,None]*ar+(mu@ar.T)[:,None]*br);bias=true_at_mu-student_at_mu-linear@mu;repaired=student+x@linear.T+bias
    # Independently evaluate the centered quadratic error that affine repair must leave.
    dx=x-mu;direct_delta=((dx@ar.T)*(dx@br.T))@writer.T-((dx@L.T)*(dx@R.T))@(C0/cscale).T;replay=float(((repaired-teacher)-direct_delta).norm()/direct_delta.norm());assert replay<1e-9
    def scores(value):return dict(total_error=float((value-teacher).norm()/teacher.norm()),centered_error=float(((value-value.mean(0))-(teacher-teacher.mean(0))).norm()/(teacher-teacher.mean(0)).norm()),mean_error=float((value.mean(0)-teacher.mean(0)).norm()/teacher.mean(0).norm()))
    programs[geometry]=dict(a=ar,b=br,writer=torch.linalg.solve_triangular(RU,writer*cscale,upper=True),linear=torch.linalg.solve_triangular(RU,linear*cscale,upper=True),bias=torch.linalg.solve_triangular(RU,(bias*cscale)[:,None],upper=True).flatten())
    rows.append(dict(geometry=geometry,policy=policy,width=width,coefficient_error=error,normal_residual=residual,raw=scores(student),affine_repaired=scores(repaired),affine_identity_replay=replay,reduced_factor_coefficients=3*1152*width,extra_affine_coefficients=1152*1152+1152,parameter_saving_with_repair=1-(3*1152*width+1152*1152+1152)/(3*1152*4608)));print(json.dumps(rows[-1]),flush=True)
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 import torch.nn.functional as F
 model=Bilin18TorchBackend.load('cuda').model.float();evaluation=[];replays=[]
 def logits(residual):return 30*torch.tanh(model.lm_head(F.rms_norm(residual,(1152,)))/30)
 for panel in (1,2):
  tokens=torch.load(P/f'FRONTIER_FRESH_TOKENS_V{panel}.pt',weights_only=True)
  for domain in ('fineweb','stdlib'):
   totals={name+'_'+arm:dict(error=0.,truth=0.,ce=0.,n=0) for name in programs for arm in ('raw','affine')}
   for row in tokens[domain]:
    cache=capture(model,row[None,:256].cuda());h=cache['h17'];z=F.rms_norm(h,(1152,));native_poly=model.transformer.h[17].mlp(z)-model.transformer.h[17].mlp.Down_bias;final=cache['final'];replays.append(float((h+native_poly+model.transformer.h[17].mlp.Down_bias-final).norm()/final.norm()))
    truth_logits=logits(final)[:,16:255];background=final-native_poly;base_logits=logits(background)[:,16:255];effect=truth_logits-base_logits;labels=row[17:256].cuda()[None];native_ce=F.cross_entropy(truth_logits.flatten(0,1),labels.flatten(),reduction='sum')
    for name,p in programs.items():
     zz=z.double();raw=((zz@p['a'].T)*(zz@p['b'].T))@p['writer'].T
     for arm in ('raw','affine'):
      replacement=raw if arm=='raw' else raw+zz@p['linear'].T+p['bias'];changed=logits(background+replacement.float())[:,16:255];error=(changed-truth_logits).double().square().sum();true=effect.double().square().sum();ce=F.cross_entropy(changed.flatten(0,1),labels.flatten(),reduction='sum')-native_ce;v=totals[name+'_'+arm];v['error']+=float(error);v['truth']+=float(true);v['ce']+=float(ce);v['n']+=labels.numel()
   evaluation.extend(dict(panel=panel,domain=domain,candidate=k,effect_relative_error=(v['error']/v['truth'])**.5,ce_added=v['ce']/v['n'],tokens=v['n']) for k,v in totals.items())
 pred=dict(pred_a_instrument=max(replays)<1e-5 and all(r['affine_identity_replay']<1e-9 for r in rows),pred_b_endpoint=all(r['effect_relative_error']<.1 for r in evaluation if r['candidate'].endswith('_affine')),pred_c_repair=all(r['effect_relative_error']<next(t['effect_relative_error'] for t in evaluation if t['panel']==r['panel'] and t['domain']==r['domain'] and t['candidate']==r['candidate'].replace('_affine','_raw')) for r in evaluation if r['candidate'].endswith('_affine')))
 (P/'FULL_CHANNEL_NATIVE_V1.json').write_text(json.dumps(dict(calibration=rows,evaluation=evaluation,predictions=pred,maximum_native_replay=max(replays),seconds=time.monotonic()-start,scope='Previously opened two panels,64FineWeb+32code documents,positions16:254. Full lastMLP polynomial replacement, Downbias preserved, native finalRMS/unembedding/softcap. Error normalized by native fullMLP effect versus polynomial-removed background. No new text fitting or independent fresh validation; no semantic circuit assertion.'),indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
