import itertools,json,time
from pathlib import Path
import numpy as np
import torch
from check_trainable_dag import planted
P=Path(__file__).resolve().parent

def rule(n):
 nodes,weights=np.polynomial.hermite.hermgauss(n);indices=torch.cartesian_prod(*[torch.arange(n)]*4);x=torch.tensor(nodes*2**.5)[indices];w=torch.tensor(weights/np.pi**.5)[indices].prod(1);return torch.cat([x,torch.ones(len(x),1)],1),w

def features(x,U):
 a=x@U.T;q=a[:,0]*a[:,1]+a[:,2]*a[:,3];return torch.stack([q.square(),torch.ones_like(q)],1)

def writer(phi,y,w):
 z=phi[:,0];zm=(w*z).sum();ym=(w[:,None]*y).sum(0);variance=(w*(z-zm).square()).sum();slope=(w[:,None]*(z-zm)[:,None]*(y-ym)).sum(0)/variance.clamp_min(1e-24);return torch.stack([slope,ym-slope*zm],1),variance

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);d,out=planted('square_quadratic');linear=[n for n in d.reachable(out) if d.nodes[n][0]=='linear'];U=torch.tensor([[float(c) for _,c in d.nodes[n][1]] for n in linear[:4]]);C=torch.tensor([[float(c) for _,c in d.nodes[n][1]] for n in out])
 # Graph canonical reference order puts constant before q² in each output.
 C=C.flip(1);x,w=rule(5);x7,w7=rule(7);y=features(x,U)@C.T;y7=features(x7,U)@C.T;replay=float((y-d.evaluate(out,x[:,:4])).norm()/y.norm());assert replay<1e-10
 torch.manual_seed(202);xv=torch.cat([torch.randn(4096,4),torch.ones(4096,1)],1);yv=features(xv,U)@C.T;den=(w[:,None]*y.square()).sum();den7=(w7[:,None]*y7.square()).sum();quadrature=float(abs(den-den7)/den);records=[]
 for optname,rate,seed,mode in itertools.product(['adam','muon'],[.005,.03],[0,1],['joint','analytic']):
  torch.manual_seed(seed);u=torch.nn.Parameter(torch.randn(4,5)*.3);c=torch.nn.Parameter(torch.randn(2,2)*.3);params=[u,c] if mode=='joint' else [u];opt=torch.optim.Adam(params,lr=rate) if optname=='adam' else torch.optim.Muon(params,lr=rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');saved=None;floor_hits=0
  for step in range(1500):
   opt.zero_grad();phi=features(x,u)
   if mode=='analytic':readout,variance=writer(phi,y,w);floor_hits+=int(float(variance.detach())<1e-24)
   else:readout=c
   loss=(w[:,None]*(phi@readout.T-y).square()).sum()/den
   if float(loss.detach())<best:best=float(loss.detach());saved=(u.detach().clone(),readout.detach().clone())
   loss.backward();opt.step()
  with torch.no_grad():
   finalphi=features(x,u);finalc=writer(finalphi,y,w)[0] if mode=='analytic' else c;final=float((w[:,None]*(finalphi@finalc.T-y).square()).sum()/den)
   if final<best:best=final;saved=(u.detach().clone(),finalc.detach().clone())
   su,sc=saved;exact7=float((w7[:,None]*(features(x7,su)@sc.T-y7).square()).sum()/den7);validation=float((features(xv,su)@sc.T-yv).norm()/yv.norm());discrepancy=abs(best-exact7);quadrature=max(quadrature,discrepancy)
  row=dict(optimizer=optname,rate=rate,seed=seed,writer=mode,best_exact_relative_error=best**.5,final_exact_relative_error=final**.5,fresh_probe_relative_error=validation,quadrature_squared_error_discrepancy=discrepancy,variance_floor_hits=floor_hits);records.append(row);print(row,flush=True)
 recovery={mode:sum(r['best_exact_relative_error']<.01 for r in records if r['writer']==mode) for mode in ['joint','analytic']};predictions=dict(pred_a=replay<1e-10 and quadrature<1e-10 and all(np.isfinite(r['final_exact_relative_error']) for r in records),pred_b=min(r['best_exact_relative_error'] for r in records if r['writer']=='analytic')<.001,pred_c=recovery['analytic']>=recovery['joint']);result=dict(records=records,teacher_replay=replay,quadrature_discrepancy=quadrature,recoveries=recovery,predictions=predictions,seconds=time.perf_counter()-start);(P/'DAG_SQUARE_OPTIMIZER_V1.json').write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',recovery,predictions,flush=True)
if __name__=='__main__':main()
