"""Conditional leave-one-chunk-out ridge selection; fixed graph is not cross-fitted."""
import json,hashlib
from pathlib import Path
import torch
from pairwise_reader_graph import source_reads
P=Path(__file__).resolve().parent

def ridge(F,e,lam):
 if lam is None:return torch.zeros(F.shape[1],dtype=F.dtype)
 scale=F.square().mean(0).sqrt().clamp_min(1e-20);X=F/scale
 # Augmented least squares avoids squaring conditioning at lambda zero.
 if lam>0:
  X=torch.cat((X,(len(F)*lam)**.5*torch.eye(F.shape[1],dtype=F.dtype)))
  e=torch.cat((e,torch.zeros(F.shape[1],dtype=F.dtype)))
 return torch.linalg.lstsq(X,-e,driver='gelsd').solution/scale

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 plan=json.loads((P/'READ_UPDATE_CV_PLAN_V1.json').read_text())
 d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);g=torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True)
 z,h=d['z'],d['h'];s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();pair=d['pairs'][2]
 true=torch.stack([(z@q*z).sum(-1) for q in pair['Qs']],-1);reads=source_reads(z,g)[:,4:6];carry=h@pair['a']-true[:,0]
 first=(carry+.5*reads[:,0])/s-pair['alpha'];target=((carry+.5*true[:,0])/s-pair['alpha'])*(true[:,1]/s-pair['beta']);error=first*(reads[:,1]/s-pair['beta'])-target
 V=g['pairs']['2']['private_reader'][:,-32:];V=V/V.norm(dim=0);psi=((z-d['mu'])@V).square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V);F=first[:,None]/s[:,None]*psi
 ids=d['indices'];chunks=(ids//64).unique();fitmask=torch.zeros(len(z),dtype=torch.bool);fitmask[ids]=True
 records=[]
 for lam in plan['lambdas']:
  folds=[]
  for chunk in chunks:
   test=fitmask&(torch.arange(len(z))//64==chunk);train=fitmask&~test
   weights=ridge(F[train],error[train],lam);pred=error[test]+F[test]@weights
   folds.append(dict(chunk=int(chunk),error_energy=float(pred.square().sum()),original_error_energy=float(error[test].square().sum())))
  records.append(dict(lam=lam,folds=folds,pooled_error_energy=sum(r['error_energy'] for r in folds)))
 # Grid ordered zero update first; deterministic first-minimum tie handling.
 selected=min(range(len(records)),key=lambda i:records[i]['pooled_error_energy']);lam=records[selected]['lam'];weights=ridge(F[fitmask],error[fitmask],lam)
 result=[]
 for name,mask in [('fitted',fitmask),('other_historical',~fitmask)]:
  e=error[mask];pred=e+F[mask]@weights;den=(target[mask]-target[mask].mean()).norm()
  result.append(dict(group=name,before=float(e.norm()/den),after=float(pred.norm()/den),ratio=float(pred.norm()/e.norm())))
 # Successor diagnostic: how many directions survive the selected regularization?
 scale=F[fitmask].square().mean(0).sqrt().clamp_min(1e-20);X=F[fitmask]/scale
 eigenvalues=torch.linalg.eigvalsh(X.T@X/len(X)).clamp_min(0)
 effective_df=0. if lam is None else float((eigenvalues/(eigenvalues+lam)).sum()) if lam>0 else int((eigenvalues>1e-10*eigenvalues.max()).sum())
 fold_optima=[]
 for j,chunk in enumerate(chunks):
  best=min(range(len(records)),key=lambda i:records[i]['folds'][j]['error_energy'])
  fold_optima.append(dict(chunk=int(chunk),best_lambda=records[best]['lam']))
 # Independent read-level execution verifies affine error/design construction.
 changed_b=reads[:,1]+psi@weights;actual=first*(changed_b/s-pair['beta'])-target;replay=float((actual-(error+F@weights)).norm()/actual.norm());assert replay<1e-10
 baseline=records[0]['pooled_error_energy'];cv_ratio=(records[selected]['pooled_error_energy']/baseline)**.5
 out=dict(effective_degrees_of_freedom=effective_df,design_eigenvalues=eigenvalues.tolist(),fold_optima=fold_optima,plan=plan,selection_index=selected,selected_lambda=lam,cv_error_ratio=cv_ratio,records=records,final_scores=result,delta_weights=weights.tolist(),execution_replay=replay,predictions=dict(cv_improves_5pct=cv_ratio<=.95,other_historical_no_worse=result[1]['ratio']<=1),scope='Conditional coefficient selection only: graph, feature directions, affine correction and metric were already constructed using historical data. Leave-one-chunk-out is not independent full-pipeline validation; no document independence, no exported/adopted replacement. Original safeguards not imposed during folds, so no fidelity-pass claim.')
 (P/'READ_UPDATE_CV_V1.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ('selected_lambda','cv_error_ratio','final_scores','execution_replay','predictions')},indent=2))
if __name__=='__main__':main()
