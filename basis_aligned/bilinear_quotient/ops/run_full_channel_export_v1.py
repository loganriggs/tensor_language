#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_export pred_c_price
"""Export frozen covariance full-layer raw/affine program;0nativecaptures.
FP32 relative drift<1e-4; literal storedfloats14,067,072."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from channel_deletion_cost import costs,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,widths=[3686],captures=0)));return
 torch.set_grad_enabled(False);start=time.monotonic();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 C0=RU@w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];O=w('transformer.h.17.attn.c_proj.weight');S=torch.linalg.cholesky(torch.eye(1152,device='cuda',dtype=torch.float64)+D@D.T+O@O.T)
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h=data['h'].cuda().double();h=h/(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();mu=h.mean(0);x=h.clone();h-=mu;ev,V=torch.linalg.eigh(h.T@h/len(h));cov=(V*ev.clamp_min(1e-8*ev.mean()).sqrt())@V.T;rows=[];programs={}
 for geometry,root in [('activation_covariance',cov)]:
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

 program={k:v.float().cpu().contiguous() for k,v in programs['activation_covariance'].items()};original=programs['activation_covariance'];xx=x.double();truth=((xx@original['a'].T)*(xx@original['b'].T))@original['writer'].T+xx@original['linear'].T+original['bias'];p32={k:v.cuda() for k,v in program.items()};xx32=x.float();actual=((xx32@p32['a'].T)*(xx32@p32['b'].T))@p32['writer'].T+xx32@p32['linear'].T+p32['bias'];drift=float((actual.double()-truth).norm()/truth.norm());count=sum(v.numel() for v in program.values());assert count==14067072 and drift<1e-4
 torch.save(program,P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt')
 import hashlib
 out=dict(calibration=rows,float32_relative_drift=drift,stored_floats=count,program_sha256=hashlib.sha256((P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt').read_bytes()).hexdigest(),predictions=dict(pred_a_instrument=all(r['affine_identity_replay']<1e-9 for r in rows),pred_b_export=drift<1e-4,pred_c_price=count==14067072),scope='Frozen standalone lastMLP polynomial residual writer given normalized1152input. Raw variant omits linear/bias. Original Downbias remains external. No QR frame,teacherweights orcalibration needed for execution after normalizedinput supplied.')
 (P/'FULL_CHANNEL_EXPORT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
