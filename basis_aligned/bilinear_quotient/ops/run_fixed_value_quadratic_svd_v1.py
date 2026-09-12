#!/usr/bin/env python3
# BQGATE:0bodyforwards;head9.8;2targets x2seeds;rank8;120seconds each;600seconds wall.
"""pred_a all singular-pair residuals<=1e-6 with live leading singular values.
pred_b rank8 captured coefficient fraction>=.50 for both targets/all seeds.
pred_c seed singular-value relative differences<=1e-6 for both targets.
Null: declared quadratic blocks insufficient or iterative solver unconverged.
No native causal promotion; full QK norms/background remain external.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch,numpy as np
from scipy.sparse.linalg import LinearOperator,svds,ArpackNoConvergence
from fixed_value_quadratic_operator_v1 import metric_power
from fixed_value_quadratic_energy_v1 import energy
from folded_normalized_router_v1 import rotary
STEM='FIXED_VALUE_QUADRATIC_SVD_V1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 assert json.loads((P/'FIXED_VALUE_QUADRATIC_ENERGY_V1_CONTROL.json').read_text())['pred_'+'a']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('0bodyforwards;4rank8 matrix-free SVDs;120seconds each');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(600);torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');q1,k1,q2,k2=[p[n][1].cuda().double() for n in ['q1','k1','q2','k2']];v=p['current_value_readers'][1].cuda().double();bs=torch.linalg.qr(torch.cat((k1,k2,v[None])).T,mode='reduced').Q;bq=torch.linalg.qr(torch.cat((q1,q2)).T,mode='reduced').Q;v=bs.T@v;n=bs.shape[1];m=bq.shape[1];band=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'][1,0].cuda().double();rows=[];programs=[]
 def pairs(mode,pos):
  r=rotary(8,128).cuda().T@rotary(pos,128).cuda();a=(q1@bq).T@r;b=(q2@bq).T@r
  if mode=='full':return [(a@(k1@bs),b@(k2@bs))]
  ia=(k1@band)@(band.T@bs);ib=(k2@band)@(band.T@bs);return [(a@ia,b@ib),(a@(k1@bs-ia),b@(k2@bs-ib))]
 def apply(x,ps):
  x=(x+x.T)/2;z=metric_power(x,v,.5);y=sum(a@z@b.T for a,b in ps);return (y+y.T)/2
 def adjoint(y,ps):
  y=(y+y.T)/2;z=sum(a.T@y@b for a,b in ps);return metric_power((z+z.T)/2,v,.5)
 for mode in ['full','even']:
  ps=pairs(mode,7);held=pairs(mode,0);total=float(energy(ps,v));heldtotal=float(energy(held,v));assert total>1e-20
  for seed in [17,29]:
   began=time.perf_counter();calls=[0,0]
   def mv(x):
    if time.perf_counter()-began>120:raise TimeoutError('120second matrix-free SVD budget')
    calls[0]+=1;return apply(torch.as_tensor(np.asarray(x).reshape(n,n),device='cuda'),ps).cpu().numpy().ravel()
   def rmv(y):
    if time.perf_counter()-began>120:raise TimeoutError('120second matrix-free SVD budget')
    calls[1]+=1;return adjoint(torch.as_tensor(np.asarray(y).reshape(m,m),device='cuda'),ps).cpu().numpy().ravel()
   op=LinearOperator((m*m,n*n),matvec=mv,rmatvec=rmv,dtype=np.float64)
   try:
    u,s,vh=svds(op,k=8,which='LM',tol=1e-9,maxiter=500,v0=np.random.default_rng(seed).standard_normal(min(op.shape)));order=np.argsort(s)[::-1];s=s[order];u=u[:,order];vh=vh[order];right=torch.from_numpy(vh.copy()).cuda().reshape(8,n,n);left=torch.from_numpy(u.T.copy()).cuda().reshape(8,m,m);sigma=torch.from_numpy(s.copy()).cuda();res=[]
    for i in range(8):res.append(max(float((apply(right[i],ps)-sigma[i]*left[i]).norm()/sigma[0]),float((adjoint(left[i],ps)-sigma[i]*right[i]).norm()/sigma[0])))
    heldcapture=sum(float(apply(z,held).square().sum()) for z in right);row=dict(target=mode,seed=seed,termination='svds_returned',singular_values=s.tolist(),max_singular_residual=max(res),target_energy=total,captured_fraction=float(np.square(s).sum()/total),held_target_energy=heldtotal,held_captured_fraction=heldcapture/heldtotal,seconds=time.perf_counter()-began,operator_calls=calls)
    programs.append(dict(target=mode,seed=seed,source_basis=bs.cpu(),query_basis=bq.cpu(),value=v.cpu(),quadratics=metric_power(right,v,-.5).cpu(),singular_values=sigma.cpu()))
   except (TimeoutError,ArpackNoConvergence) as stopped:row=dict(target=mode,seed=seed,termination=str(stopped),seconds=time.perf_counter()-began,operator_calls=calls)
   rows.append(row);print(json.dumps(row),flush=True);(P/(STEM+'_PROGRESS.json')).write_text(json.dumps(rows,indent=2)+'\n')
 complete=[r for r in rows if r['termination']=='svds_returned'];agreements=[]
 for mode in ['full','even']:
  rr=[r for r in complete if r['target']==mode]
  if len(rr)==2:agreements.append(float(np.linalg.norm(np.array(rr[0]['singular_values'])-rr[1]['singular_values'])/np.linalg.norm(rr[0]['singular_values'])))
 A=len(complete)==4 and all(r['max_singular_residual']<=1e-6 and r['singular_values'][0]>1e-12 for r in complete)
 result={'pred_a':A,'pred_b':A and all(r['captured_fraction']>=.5 for r in complete),'pred_c':A and len(agreements)==2 and max(agreements)<=1e-6,'rows':rows,'seed_singular_value_relative_errors':agreements,'seconds':time.perf_counter()-tic,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scope':'Head9.8 fixed current-value; fulljointkey and earlierselected even numerator. Rank8 weighted quadratic blocks at query8/source7; frozen sourcequadratics rescored atsource0. Iterative residuals and two-seed agreement are numerical evidence, not proof no leading mode was missed. No native normalization or causal validation; no CP rank/budget equivalence.'}
 torch.save(programs,art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
