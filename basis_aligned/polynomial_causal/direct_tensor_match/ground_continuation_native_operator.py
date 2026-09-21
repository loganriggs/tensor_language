"""Ground a learned continuation term against the original folded scalar operator.

pred_a: native saved-target replay<1e-5; scalar matrix/factor replay<1e-10.
pred_b: frozen product approximates the original scalar observer within.25 in
independent centered calibration covariance norm.
pred_c: frozen error <=1.10 globally optimal rank-one error in the same metric.
Observer is fixed by the learned writer and vocabulary-centered metric; all
other fitted terms are exposed as background, not silently discarded.
"""
from pathlib import Path
import json,time,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
out=p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.json';assert not out.exists()
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');ru=torch.linalg.qr(state['lm_head.weight'].double(),mode='r').R
L=state['transformer.h.17.mlp.Left.weight'].double();R=state['transformer.h.17.mlp.Right.weight'].double();C=ru@state['transformer.h.17.mlp.Down.weight'].double()
e=torch.load(p/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt',weights_only=True)['half0'];parent=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];ids=torch.load(p/'MIDPOINT_JOINT_METRIC_STABILITY_V1.pt',weights_only=True)['ids']
rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();y=rows['y'].flatten(0,1).double();S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
w=e['W'][:,2];q=S.T@(S@w)/((S@w).square().sum());assert abs(float(q@w)-1)<1e-12
channel=C.T@q;matrix=L.T@(channel[:,None]*R)+R.T@(channel[:,None]*L)
full=((n@L.T)*(m@R.T)+(n@R.T)*(m@L.T))@C.T
native_replay=float((full-y).norm()/y.norm());scalar=((n@matrix)*m).sum(1);scalar_factor=full@q;scalar_replay=float((scalar-scalar_factor).norm()/scalar_factor.norm());assert native_replay<1e-5 and scalar_replay<1e-10
nb=n.mean(0);mb=m.mean(0);nc=n-nb;mc=m-mb;Mn=nc.T@nc/len(n);Mm=mc.T@mc/len(m)
roots=[];inverses=[];discard=[]
for M in [Mn,Mm]:
 ev,V=torch.linalg.eigh(M);mask=ev>1e-10*ev.max();cut=ev.clamp_min(0)*mask;root=(V*cut.sqrt())@V.T;inverse=(V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T;roots.append(root);inverses.append(inverse);discard.append(float(ev.clamp_min(0)[~mask].sum()/ev.clamp_min(0).sum()))
N,M=roots;weighted=N@matrix@M;U,s,Vh=torch.linalg.svd(weighted,full_matrices=False)
a=inverses[0]@U[:,0]*s[0].sqrt();b=inverses[1]@Vh[0]*s[0].sqrt();optimal=torch.outer(a,b);frozen=torch.outer(e['A'][:,2],e['B'][:,2])
# Project the remaining approximate centered interaction through this same reader.
projected_parent=(parent['A']*(q@parent['reduced_writers'])[None,:])@parent['B'].T
old_local=(parent['A'][:,ids]*(q@parent['reduced_writers'][:,ids])[None,:])@parent['B'][:,ids].T
new_local=(e['A']*(q@e['W'])[None,:])@e['B'].T
background=projected_parent-old_local+new_local-frozen
norm=lambda t:float((N@t@M).norm())
target_norm=norm(matrix);records=[]
for label,t in [('frozen_product',frozen),('optimal_native_rank1',optimal),('frozen_with_background',frozen+background),('background_alone',background)]:
 pair=((nc@t)*mc).sum(1);truth=((nc@matrix)*mc).sum(1)
 records.append(dict(candidate=label,weighted_coefficient_error=norm(t-matrix)/target_norm,weighted_norm_over_target=norm(t)/target_norm,paired_calibration_error=float((pair-truth).norm()/truth.norm())))
err=records[0]['weighted_coefficient_error'];best=records[1]['weighted_coefficient_error'];svderror=float(s[1:].norm()/s.norm());assert abs(best-svderror)<1e-8
residual=matrix-background;coeffcos=float(((N@frozen@M)*(N@optimal@M)).sum()/((N@frozen@M).norm()*(N@optimal@M).norm()))
result=dict(predictions=dict(pred_a_instrument=native_replay<1e-5 and scalar_replay<1e-10,pred_b_native_grounding=err<=.25,pred_c_near_optimal=err<=1.10*best),records=records,native_replay=native_replay,scalar_matrix_replay=scalar_replay,discarded_covariance_trace=discard,rank1_lower_bound_error=svderror,frozen_to_native_optimal_coefficient_cosine=coeffcos,background_relative_norm=norm(background)/target_norm,frozen_residual_error=norm(frozen-residual)/norm(residual),top_singular_energy_fractions=(s[:8].square()/s.square().sum()).tolist(),seconds=time.perf_counter()-start,scope='Original trained last-MLP weights, fixed learned output observer q=K w/(wTKw). Centered intermediate-input bilinear scalar, not full model. Rank-one optimum is exact for retained separable covariance metric; trace cutoff quantified. Other fitted terms explicitly included as background. Paired rows are calibration, not fresh/OOD.')
torch.save(dict(q=q,native_matrix=matrix,frozen_a=e['A'][:,2],frozen_b=e['B'][:,2],native_rank1_a=a,native_rank1_b=b,writer=w,mean_n=nb,mean_m=mb,background_matrix=background,R_U=ru),p/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
