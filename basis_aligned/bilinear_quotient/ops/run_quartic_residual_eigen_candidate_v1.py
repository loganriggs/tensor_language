#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4residualdirections,8eigenmatrices,128cached validationvectors.
"""Native operator price passed; bound config caps200actions/direction,600sec total.
pred_a all eigen residuals<=1e-5, baseline/Schur replay<=1e-8 and descent;
pred_b capture gain>=1%; pred_c native write error reduced>=20%.
Frozen LBFGS V1,32rank16quadratics,592704floats. No nonlinear refit/textfit.
"""
import os,sys,json,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from coupled_quartic_writer_v1 import gram,target_cross,solve,features
from quartic_weighted_trace_v1 import producer_core,native_action,fitted_action
from quartic_matrixfree_eigen_v2 import eigenmatrices
from quartic_residual_addition_v1 import scores
from composed_quartic_contraction_v1 import contract
STEM='QUARTIC_RESIDUAL_EIGEN_CANDIDATE_V1'
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text());files=binding['files'];config=binding['config']
 assert all(digest(k)==v for k,v in files.items())
 price=json.loads((P/'QUARTIC_EIGEN_OPERATOR_PRICE_V1_RESULT.json').read_text())
 assert price['pred_a'] and price['pred_b'] and price['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,residual_directions=4,maximum_eigenpairs=8,maximum_operator_actions=4*(config['max_actions']+2),fitted_floats=592704)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(config['alarm_seconds'])
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 p=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)
 full=torch.load(P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_PROGRAMS.pt',weights_only=True)['programs'][0]
 assert p['seed']==full['seed']==11511
 assert torch.equal(p['input_readers'],full['input_readers']) and torch.equal(p['inner_weights'],full['inner_weights']) and torch.equal(p['output_writers'],full['output_writers'])
 b=p['input_readers'].cuda();n=p['inner_weights'].cuda();writer=p['output_writers'].cuda()
 diagids=(full['pairs'][0]==full['pairs'][1]).nonzero().flatten()
 c=full['target_cross'][diagids].cuda();k=gram(b,n);mix,diagnostic=solve(k,c)
 capture=float((mix*c).sum());old=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_RESULT.json').read_text())
 replay=abs(capture/p['divisor']+old['history'][-1]['objective']);assert replay<=1e-8
 weak=int((mix.square().sum(-1)/torch.linalg.inv(k).diagonal()).argmin())
 retained=torch.tensor([j for j in range(len(b)) if j!=weak],device='cuda')
 retained_mix,_=solve(k[retained][:,retained],c[retained])
 state=torch.load(next(k for k in files if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 l,r,down,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0]);output=(metric@writer).T@d1
 cores=torch.stack([producer_core(down,l1,r1,row,scale) for row in output])
 directions=torch.tensor([[1.,0.],[0.,1.],[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]],device='cuda')
 bank=[];weights=[];reports=[]
 for index,direction in enumerate(directions):
  h=torch.einsum('m,mab->ab',direction,cores);a=retained_mix@direction
  def action(q):return native_action(l,r,h,q)-fitted_action(b[retained],n[retained],a,q)
  values,matrices,report=eigenmatrices(action,1152,device='cuda',k=2,seed=91961+index,tol=1e-6,ncv=12,maxiter=100,max_actions=config['max_actions'])
  report['direction']=direction.cpu().tolist();report['eigenvalues']=values.tolist();reports.append(report);print(json.dumps(report),flush=True)
  if report['status']!='converged' or len(matrices)!=2 or any(not math.isfinite(e) or e>1e-5 for e in report['relative_eigen_residuals']):
   out.write_text(json.dumps(dict(pred_a=False,pred_b=False,pred_c=False,stage='eigensolver_incomplete',eigen_reports=reports,scope='No valid candidate program; do not interpret as absent structure.'),indent=2)+'\n');return
  for q in matrices:
   values,frame=torch.linalg.eigh(q.cpu());ids=values.abs().argsort()[-16:]
   bank.append(frame[:,ids].cuda());weights.append((values[ids]/values[ids].norm()).cuda())
 newb=torch.stack(bank);newn=torch.stack(weights)
 cc=target_cross(newb,newn,lambda slots:contract(slots,l,r,down,l1,r1,output,scale))
 allb=torch.cat([b,newb]);alln=torch.cat([n,newn]);allc=torch.cat([c,cc]);allk=gram(allb,alln)
 gain,variance,base=scores(allk,allc,retained);chosen=int(gain.argmax())
 selected=torch.arange(len(b),device='cuda');selected[weak]=chosen
 mixing,diagnostic=solve(allk[selected][:,selected],allc[selected]);captured=float((mixing*allc[selected]).sum())
 schur=abs(captured-base-float(gain[chosen]))/capture
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda();den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].cuda()
 def error(bb,nn,aa):return float((features(bb,nn,x)@aa@writer.T/den[:,None]-ref).norm()/ref.norm())
 before=error(b,n,mix);after=error(allb[selected],alln[selected],mixing)
 torch.save(dict(seed=11511,input_readers=allb[selected].cpu(),inner_weights=alln[selected].cpu(),mixing=mixing.cpu(),output_writers=writer.cpu(),divisor=p['divisor']),ap)
 result={'pred_a':max(replay,schur,diagnostic['normal_residual'])<=1e-8 and captured>=capture*(1-1e-10),'pred_b':captured>=1.01*capture,'pred_c':after<=.8*before}
 result.update(dict(eigen_reports=reports,dropped_node=weak,chosen_candidate=chosen,coefficient_gain_fraction=captured/capture-1,baseline_replay_error=replay,schur_identity_error=schur,
   initial_native_error=before,final_native_error=after,artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
   scope='Four fixed output directions, residual eigenmatrices rank16-truncated then Schur-selected. Oldnode fallback, frozen square baseline; no jointrefit/datafit/global or circuit claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
