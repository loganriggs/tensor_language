"""Joint input dictionary for two frozen quadratic source forms; CPU screen.
Target rank16: <=1.10 separate16 product variation error, >=30%stored reduction.
Full32-support replay<1e-8. No native adoption or data-free metric claim.
"""
from pathlib import Path
import torch,json
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
e=torch.load(p/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt',weights_only=True)['covariance_16'];cal=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);z=cal['z'].flatten(0,1).double();mu=e['mu'];delta=z-mu;cov=delta.T@delta/len(delta);v,U=torch.linalg.eigh(cov);mask=v>v.max()*1e-10;root=(U*v.clamp_min(0).sqrt())@U.T;inv=(U*torch.where(mask,v.clamp_min(1e-30).rsqrt(),0))@U.T
Qs=[(e[k+'_reader']*e[k+'_eigenvalues'])@e[k+'_reader'].T for k in ['a','b']];Ms=[root@Q@root for Q in Qs];gram=sum(M@M.T for M in Ms);g,P=torch.linalg.eigh(gram);P=P[:,g.argsort(descending=True)];scale=cal['recipient_scale'].flatten().double();hread=cal['h_reader'].flatten().double();fold=torch.load(p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);truth=torch.stack([((z@fold[k]['matrix'])*z).sum(1) for k in ['a','b']],1)
phi=lambda q:((hread-.5*q[:,0])/scale-e['alpha'])*(q[:,1]/scale-e['beta'])
target=phi(truth);den=(target-target.mean()).norm();sep=torch.stack([e[k+'_constant']+delta@e[k+'_linear']+((delta@Q)*delta).sum(1)-torch.trace(cov@Q) for k,Q in zip(['a','b'],Qs)],1);sep_error=float((phi(sep)-target).norm()/den);records=[];exports={}
for rank in [8,16,24,32]:
 V=P[:,:rank];shared=inv@V;center=mu@shared;coordinates=delta@shared;program=dict(shared_reader=shared.clone(),h_reader=e['a'].clone(),residual_writer=torch.linalg.solve(e['R_U'],e['writer']),alpha=e['alpha'].clone(),beta=e['beta'].clone());pred=[]
 for k,M in zip(['a','b'],Ms):
  core=V.T@M@V;ev,W=torch.linalg.eigh(core);order=ev.abs().argsort(descending=True)[:min(rank,16)];ev=ev[order];W=W[:,order];core=(W*ev)@W.T
  quadratic=((coordinates@core)*coordinates).sum(1);qmean=torch.trace((shared.T@cov@shared)@core);pred.append(e[k+'_constant']+delta@e[k+'_linear']+quadratic-qmean)
  program[k+'_inner_reader']=W;program[k+'_eigenvalues']=ev;program[k+'_linear']=e[k+'_linear']-2*shared@(core@center);program[k+'_bias']=e[k+'_constant']-mu@e[k+'_linear']+center@core@center-qmean
 pred=torch.stack(pred,1);records.append(dict(shared_rank=rank,product_variation_error=float((phi(pred)-target).norm()/den),relative_to_separate= float((phi(pred)-target).norm()/den)/sep_error,source_variation_errors=[float((pred[:,j]-truth[:,j]).norm()/(truth[:,j]-truth[:,j].mean()).norm()) for j in [0,1]],separate_program_replay=float((pred-sep).norm()/sep.norm()),stored_scalars=sum(t.numel() for t in program.values()),source_squares=sum(program[k+'_eigenvalues'].numel() for k in ['a','b']),shared_input_projections=rank,additional_inner_coefficients=sum(program[k+'_inner_reader'].numel() for k in ['a','b'])));exports[str(rank)]=program
r16=records[1];result=dict(predictions=dict(pred_a_replay=records[-1]['separate_program_replay']<1e-8,pred_b_fidelity=r16['relative_to_separate']<=1.10,pred_c_storage=r16['stored_scalars']<=.7*41508),records=records,separate16_product_variation_error=sep_error,scope='CPU calibration screen. Frozen rank16 quadratic approximations projected jointly using calibration covariance. Constants/linear parts retained in centered expansion; compiled into native-z/h executor. Shared dictionary counts both small inner transforms; h,z remain native inputs.')
(p/'MIDPOINT_SOURCE_SHARED_DICTIONARY_V1.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(exports,p/'MIDPOINT_SOURCE_SHARED_DICTIONARY_V1.pt');print(json.dumps(result,indent=2))
