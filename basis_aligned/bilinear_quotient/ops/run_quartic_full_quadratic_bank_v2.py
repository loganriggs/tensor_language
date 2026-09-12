#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;4residualeigensolves,48targetmatrixactions,128cachedvalidationvectors.
"""pred_a single-full replay<=1e-8, eigen residual<=1e-5 and normal residual<=1e-8;
pred_b >=10% coefficient capture over single-full; pred_c native write error<=20%.
Eight full symmetric quadratics plus31rank16quadratics; about5.89M fitted floats.
No language-data fitting, nonlinear refit or circuit claim.
"""
import os,sys,json,time,signal,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from coupled_quartic_writer_v1 import gram,solve,features
from quartic_weighted_trace_v1 import producer_core,native_action,fitted_action
from quartic_matrixfree_eigen_v2 import eigenmatrices,SymmetricCoordinates
from quartic_full_bank_v1 import full_gram,low_full_cross
STEM='QUARTIC_FULL_QUADRATIC_BANK_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,residual_directions=4,target_matrix_actions=48,maximum_eigen_operator_actions=808)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAM.pt');assert not out.exists() and not ap.exists();signal.alarm(600)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 p=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)
 full=torch.load(P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_PROGRAMS.pt',weights_only=True)['programs'][0]
 assert p['seed']==full['seed']==11511
 assert all(torch.equal(p[k],full[k]) for k in ('input_readers','inner_weights','output_writers'))
 b=p['input_readers'].cuda();n=p['inner_weights'].cuda();writer=p['output_writers'].cuda()
 ids=(full['pairs'][0]==full['pairs'][1]).nonzero().flatten();c=full['target_cross'][ids].cuda();k=gram(b,n)
 mix,_=solve(k,c);capture=float((mix*c).sum());weak=int((mix.square().sum(-1)/torch.linalg.inv(k).diagonal()).argmin())
 retained=torch.tensor([j for j in range(32) if j!=weak],device='cuda');br=b[retained];nr=n[retained]
 kr=k[retained][:,retained];cr=c[retained];ar,_=solve(kr,cr);base=float((ar*cr).sum())
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 l,r,down,l1,r1,d1=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0]);output=(metric@writer).T@d1
 cores=torch.stack([producer_core(down,l1,r1,row,scale) for row in output])
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda();den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
 ref=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].cuda();retained_values=features(br,nr,x)
 def evaluate(q):
  q=q/q.norm();inner=br.transpose(-1,-2)@q@br
  trace=(nr*inner.diagonal(dim1=-2,dim2=-1)).sum(-1)
  cycle=(nr[:,:,None]*nr[:,None,:]*inner.square()).sum((-1,-2))
  cross=(trace.square()+2*cycle)/3
  square=q@q;diagonal=(float(q.square().sum())**2+2*float(square.square().sum()))/3
  cq=torch.stack([(q*native_action(l,r,h,q)).sum() for h in cores])
  projection=torch.linalg.solve(kr,cross);variance=diagonal-float(cross@projection)
  if variance<=1e-12:return dict(net_capture_gain_fraction=None,native_write_error=None,residual_variance=variance,excluded_as_dependent=True),None
  residual=cq-cross@ar;addition=residual.square().sum()/variance;alpha=residual/variance
  adjusted=ar-projection[:,None]*alpha[None,:]
  write=(retained_values@adjusted+((x@q)*x).sum(-1).square()[:,None]*alpha)@writer.T/den[:,None]
  error=float((write-ref).norm()/ref.norm())
  radial=float(q.trace().square()/(1152*q.square().sum()))
  row=dict(net_capture_gain_fraction=(base+float(addition))/capture-1,native_write_error=error,
           residual_variance=variance,isotropic_matrix_energy_fraction=radial)
  return row,(q,adjusted,alpha,write)
 baseline_error=float((features(b,n,x)@mix@writer.T/den[:,None]-ref).norm()/ref.norm())
 best={kind:(dict(net_capture_gain_fraction=0.,native_write_error=baseline_error),None) for kind in ('rank16','full')}
 directions=torch.tensor([[1.,0.],[0.,1.],[1/math.sqrt(2),1/math.sqrt(2)],[1/math.sqrt(2),-1/math.sqrt(2)]],device='cuda')
 rows=[];eigen_reports=[];bank=[];bank_c=[]
 for index,direction in enumerate(directions):
  h=torch.einsum('m,mab->ab',direction,cores);a=ar@direction
  def action(q):return native_action(l,r,h,q)-fitted_action(br,nr,a,q)
  values,matrices,report=eigenmatrices(action,1152,device='cuda',k=2,seed=91961+index,tol=1e-6,ncv=12,maxiter=100,max_actions=200)
  eigen_reports.append(report);assert report['status']=='converged' and len(matrices)==2
  assert all(math.isfinite(e) and e<=1e-5 for e in report['relative_eigen_residuals'])
  for local,q in enumerate(matrices):
   bank.append(q/q.norm())
   bank_c.append(torch.stack([(bank[-1]*native_action(l,r,h,bank[-1])).sum() for h in cores]))
   eigen,frame=torch.linalg.eigh(q.cpu());selected=eigen.abs().argsort()[-16:]
   truncated=((frame[:,selected]*eigen[selected])@frame[:,selected].T).cuda()
   for kind,matrix in [('rank16',truncated),('full',q)]:
    row,program=evaluate(matrix);row.update(kind=kind,direction_index=index,eigen_index=local,eigenvalue=float(values[local]),rank16_matrix_energy_fraction=float(eigen[selected].square().sum()/eigen.square().sum()))
    rows.append(row)
    if program is not None and row['net_capture_gain_fraction']>best[kind][0]['net_capture_gain_fraction']:best[kind]=(row,program)
   print(json.dumps(dict(direction=index,eigen_index=local,full_gain=rows[-1]['net_capture_gain_fraction'],rank16_gain=rows[-2]['net_capture_gain_fraction'])),flush=True)
 old=json.loads((P/'QUARTIC_FULL_QUADRATIC_CANDIDATE_V1_RESULT.json').read_text())
 replay=max(abs(best['full'][0]['net_capture_gain_fraction']-old['best_full']['net_capture_gain_fraction']),abs(best['full'][0]['native_write_error']-old['best_full']['native_write_error']))
 q=torch.stack(bank);cq=torch.stack(bank_c);cross=low_full_cross(br,nr,q);kq=full_gram(q)
 joint=torch.cat([torch.cat([kr,cross],1),torch.cat([cross.T,kq],1)],0)
 target=torch.cat([cr,cq]);weights,solver=solve(joint,target)
 captured=float((weights*target).sum());single_capture=capture*(1+old['best_full']['net_capture_gain_fraction'])
 qvalues=torch.stack([((x@matrix)*x).sum(-1).square() for matrix in q],1)
 write=(retained_values@weights[:len(br)]+qvalues@weights[len(br):])@writer.T/den[:,None]
 error=float((write-ref).norm()/ref.norm());coordinates=SymmetricCoordinates(1152,'cuda')
 packed=torch.stack([coordinates.pack(matrix) for matrix in q])
 torch.save(dict(seed=11511,input_readers=br.cpu(),inner_weights=nr.cpu(),mixing=weights[:len(br)].cpu(),full_quadratic_packed=packed.cpu(),full_quadratic_mixing=weights[len(br):].cpu(),output_writers=writer.cpu(),native_write=write.cpu(),coefficient_gram=joint.cpu(),target_cross=target.cpu()),ap)
 result={'pred_a':replay<=1e-8 and solver['normal_residual']<=1e-8,'pred_b':captured/single_capture-1>=.1,'pred_c':error<=.2}
 result.update(dict(
 single_full_replay_error=replay,solver=solver,captured_energy=captured,extra_capture_over_single_fraction=captured/single_capture-1,
 baseline_native_error=baseline_error,final_native_error=error,old_capture=capture,eigen_reports=eigen_reports,
 fitted_floats=br.numel()+nr.numel()+weights.numel()+packed.numel()+writer.numel(),validation_write_floats=write.numel(),
 artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),scope='Eight coefficient-derived full quadratic squares plus31 retained rank16 nodes, exact output refit; reused native validation only, no circuit/OOD claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True);assert result['pred_a']
if __name__=='__main__':main()
