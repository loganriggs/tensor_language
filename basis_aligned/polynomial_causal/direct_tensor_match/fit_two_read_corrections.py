"""Alternate conditionally convex readout updates, using each square in both reads."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from convex_quadratic_minimax import solve
from conditional_source_constraints import ConditionalSourceConstraints
from pairwise_reader_graph import expand
from local_shared_reader_graph import decode
from pairwise_graph_assessment import Assessment
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();plan=json.loads((P/'TWO_READ_CORRECTION_PLAN_V1.json').read_text());parent_receipt=json.loads((P/plan['parent_receipt']).read_text());receipt=parent_receipt['records'][0];edits=parent_receipt['plan']['replacements'];allocation=receipt['allocation'];offsets=np.cumsum([0]+allocation);total=sum(allocation)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);base=json.loads((P/'SOURCE_SQUARE_PLAN_V1.json').read_text())['baseline'];parent=torch.load(P/'PENCIL_JOINT_REFIT_PROGRAMS_V1.pt',weights_only=True)['calibration_shaped_inherited'];bundle=expand(parent);H0=torch.cat([decode(bundle[str(j)]) for j in range(3)]);metric=ConditionalSourceConstraints(d,H0,base);audit=Assessment(d);S=metric.S;inv=d['inverse_root'];z,h,s=metric.z,metric.h,metric.s;banks=[];features=[];atoms=[]
for j,n in enumerate(allocation):
 e,U=torch.linalg.eigh(S@(metric.true[2*j+1]-H0[2*j+1])@S);ix=e.abs().argsort(descending=True)[:14];V=inv@U[:,ix]
 for edit in edits:
  if edit['pair']==j:V[:,edit['column']]=inv@torch.tensor(edit['metric_direction'],dtype=V.dtype)
 V=V[:,:n];lam=e[ix[:n]];banks.append((V,lam));atoms.append(torch.einsum('ik,jk,k->kij',V,V,lam));centered=(z-d['mu'])@V;psi=(centered.square()-torch.einsum('ik,ij,jk->k',V,d['old_covariance'],V))*lam;gradpsi=2*centered[:,None,:]*V[None]*lam;features.append((psi,gradpsi))
X=torch.zeros(total,2,dtype=H0.dtype);X[:,1]=torch.tensor(receipt['solver']['x'],dtype=H0.dtype)
def forms(X):
 H=H0.clone()
 for j in range(3):
  for parity in (0,1):H[2*j+parity]+=torch.einsum('k,kij->ij',X[offsets[j]:offsets[j+1],parity],atoms[j])
 return H

def export(X):
 graph=copy.deepcopy(parent)
 for j,n in enumerate(allocation):
  p=graph['pairs'][str(j)];V,lam=banks[j];width=len(p['shared_indices'])+len(p['private_indices']);ids=torch.arange(width,width+n);p['private_reader']=torch.cat([p['private_reader'],V],1);p['private_indices']=torch.cat([p['private_indices'],ids]);p['product_indices']=torch.cat([p['product_indices'],torch.stack([ids,ids,torch.zeros_like(ids)])],1);p['product_weights']=torch.cat([p['product_weights'],lam[:,None]*X[offsets[j]:offsets[j+1]]])
 return audit.correct(graph)

def score_ratios(scores):return np.square([scores['native_error']/(1.1*base['native_error']),scores['covariance_error']/(1.1*base['covariance_error'])]+[a/min(.15,1.1*b) for a,b in zip(scores['per_mode_errors'],base['per_mode_errors'])]+[a/(1.1*b) for a,b in zip(scores['euclidean_jacobian_errors'],base['euclidean_jacobian_errors'])])
initial=float(metric.ratios_full(forms(X)).max());history=[];sweep_start=initial
for step in range(plan['steps']):
 parity=step%2;fixed=X.clone();fixed[:,parity]=0;H=forms(fixed);delta=metric.true-H;lin=2*delta@d['mu'];bias=torch.einsum('ij,oij->o',d['old_covariance'],delta)-torch.einsum('i,oij,j->o',d['mu'],delta,d['mu']);reads=torch.einsum('ni,oij,nj->no',z,H,z)+z@lin.T+bias;grads=2*torch.einsum('oij,nj->noi',H,z)+lin[None];values,jacs=metric.responses(H)
 G=torch.zeros(8,total,total,dtype=H.dtype);b=torch.zeros(8,total,dtype=H.dtype);c=torch.zeros(8,dtype=H.dtype)
 for j,n in enumerate(allocation):
  sl=slice(offsets[j],offsets[j+1]);T=metric.true[2*j:2*j+2];E=H[2*j:2*j+2]-T
  for k,(M,name) in enumerate(((metric.I,'native_error'),(S,'covariance_error'))):
   target=M@T@M;res=M@E@M;F=(M@atoms[j]@M).flatten(1).T;den=3*target.square().sum()*(1.1*base[name])**2;G[k,sl,sl]=F.T@F/den;b[k,sl]=F.T@res[parity].flatten()/den;c[k]+=res.square().sum()/den
  pair=d['pairs'][j];aa=(h@pair['a']-.5*reads[:,2*j])/s-pair['alpha'];bb=reads[:,2*j+1]/s-pair['beta'];psi,gradpsi=features[j]
  factor=(-.5*bb/s) if parity==0 else (aa/s);othergrad=grads[:,2*j+1] if parity==0 else grads[:,2*j]
  value_design=factor[:,None]*psi;jac_design=factor[:,None,None]*gradpsi-.5*othergrad[:,:,None]*psi[:,None,:]/s[:,None,None].square();truth=pair['truth'][d['indices']]
  entries=[(2+j,values[j]-truth,value_design,(truth-truth.mean()).norm()*min(.15,1.1*base['per_mode_errors'][j])),(5+j,jacs[j]-metric.truth_jac[j],jac_design,metric.truth_jac[j].norm()*1.1*base['euclidean_jacobian_errors'][j])]
  for k,res,F,den in entries:
   F=F.reshape(-1,n)/den;res=res.flatten()/den;G[k,sl,sl]=F.T@F;b[k,sl]=F.T@res;c[k]=res.square().sum()
 def predict(x):return np.einsum('i,kij,j->k',x,G.numpy(),x)+2*b.numpy()@x+c.numpy()
 old=X[:,parity].numpy();before=metric.ratios_full(forms(X)).numpy();assert np.max(np.abs(predict(old)-before))<1e-8
 control=fixed.clone();control[:,parity]=torch.linspace(-.3,.7,total,dtype=H.dtype);assert np.max(np.abs(predict(control[:,parity].numpy())-metric.ratios_full(forms(control)).numpy()))<1e-8
 result=solve(G.numpy(),b.numpy(),c.numpy(),initial=old);trial=X.clone();trial[:,parity]=torch.tensor(result['x'],dtype=H.dtype);direct=metric.ratios_full(forms(trial)).numpy();assert np.max(np.abs(predict(result['x'])-direct))<1e-8;accepted=direct.max()<=before.max()+1e-10
 if accepted:X=trial
 graph=export(X);scores=audit.assess(graph);replay=float(np.max(np.abs(score_ratios(scores)-metric.ratios_full(forms(X)).numpy())));assert replay<1e-8;assert scores['source_total_multiplications']==1063818
 row=dict(step=step,parity=parity,accepted=bool(accepted),maximum=float(score_ratios(scores).max()),solver=result,replay=replay,**scores);history.append(row);print(json.dumps({k:v for k,v in row.items() if k!='solver'}),flush=True)
 (P/'TWO_READ_CORRECTION_PARTIAL_V1.json').write_text(json.dumps(dict(history=history,amplitudes=X.tolist()),indent=2)+'\n')
 if row['maximum']<=1:break
 if parity==1:
  if sweep_start-row['maximum']<1e-8:break
  sweep_start=row['maximum']
out=dict(plan=plan,initial=initial,history=history,amplitudes=X.tolist(),predictions=dict(instrument=True,fidelity=history[-1]['maximum']<=1,cost=True),seconds=time.monotonic()-start)
(P/'TWO_READ_CORRECTION_V1.json').write_text(json.dumps(out,indent=2)+'\n')
