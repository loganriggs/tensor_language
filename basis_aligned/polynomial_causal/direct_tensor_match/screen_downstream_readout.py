"""Exact fixed-product readout solve for a downstream-weighted gradient surrogate.
First24cachedchunks fit; seven opened prefixes evaluate. Fixed native amplitudes.
Blend normalized coefficient error and linearized conditional Jacobian error.
"""
from pathlib import Path
import torch,json
from source_graph_metrics import export,score
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']]
root=torch.linalg.inv(d['inverse_root']);s=p['shared_mixed'];L=root@s['left_reader'];R=root@s['right_reader'];L/=L.norm(dim=0);R/=R.norm(dim=0)
ln=d['inverse_root']@L;rn=d['inverse_root']@R;delta=d['z'][:1536]-d['mu'];z=d['z'][:1536];h=d['h'][:1536];scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();xd=delta@ln;yd=delta@rn
LL=ln.T@ln;RR=rn.T@rn;LR=ln.T@rn;RL=LR.T
Kcoef=.5*((L.T@L)*(R.T@R)+(L.T@R)*(R.T@L));rank=L.shape[1]
forms=torch.stack([Q for pair in d['pairs'][:2] for Q in pair['Qs']]);reads=torch.einsum('ni,oij,nj->no',z,forms,z);objects=[]
for j,pair in enumerate(d['pairs'][:2]):
 aidx,bidx=2*j,2*j+1;u=(h@pair['a']-.5*reads[:,aidx])/scale-pair['alpha'];v=reads[:,bidx]/scale-pair['beta']
 amps=[-.5*v/scale*d['scales'][aidx],u/scale*d['scales'][bidx]]
 normalized=[forms[aidx]/d['scales'][aidx],forms[bidx]/d['scales'][bidx]]
 target=sum(a[:,None]*(2*delta@Q) for a,Q in zip(amps,normalized));energy=target.square().sum()/len(z)
 blocks=[]
 for a in amps:
  row=[]
  for b in amps:
   w=a*b;row.append((LL*(yd.T@(yd*w[:,None]))+RR*(xd.T@(xd*w[:,None]))+LR*(yd.T@(xd*w[:,None]))+RL*(xd.T@(yd*w[:,None])))/len(z))
  blocks.append(torch.cat(row,1))
 K=torch.cat(blocks,0);rhs=torch.cat([(((target@ln)*yd+(target@rn)*xd)*a[:,None]).sum(0)/len(z) for a in amps]);T=d['teacher'][2*j:2*j+2];e0=T.square().sum();bc=torch.einsum('ir,oij,jr->or',L,T,R).reshape(-1)
 # Independently materialize a few derivative design columns to verify Gram/rhs.
 small=[]
 for a in amps:
  for k in range(3):small.append((a[:,None]*(yd[:,k,None]*ln[:,k]+xd[:,k,None]*rn[:,k])).reshape(-1))
 D=torch.stack(small,1);idx=torch.tensor([0,1,2,rank,rank+1,rank+2]);gram_error=float((D.T@D/len(z)-K[idx[:,None],idx]).norm()/K[idx[:,None],idx].norm());rhs_error=float((D.T@target.reshape(-1)/len(z)-rhs[idx]).norm()/rhs[idx].norm());assert max(gram_error,rhs_error)<1e-10
 objects.append((K/energy,rhs/energy,torch.block_diag(Kcoef,Kcoef)/e0,bc/e0,gram_error,rhs_error))
records=[]
for lam in [0,.1,1,10]:
 ws=[];residuals=[];surrogate=[]
 for K,b,C,c,_,_ in objects:
  system=C+lam*K+1e-10*torch.eye(len(K),dtype=K.dtype);rhs=c+lam*b;w=torch.linalg.solve(system,rhs);ws.append(w.reshape(2,rank));residuals.append(float((system@w-rhs).norm()/rhs.norm()));surrogate.append(float((1-2*w@b+w@K@w).clamp_min(0).sqrt()))
 W=torch.cat(ws);outp=export(L,R,W,d,p);values=score(outp,d);rec=dict(lam=lam,train_linearized_relative_errors=surrogate,normal_equation_residual=max(residuals),**values);records.append(rec);print(rec,flush=True)
primary=records[2];parent=score(p,d);base=json.loads((P/'MULTIMODE_PROJECTION_SHARING_V1.json').read_text())['baseline_errors']
out=dict(records=records,parent=parent,design_replay=[dict(gram=x[4],rhs=x[5]) for x in objects],predictions=dict(pred_a_instrument=max(max(x[4:]) for x in objects)<1e-10,pred_b_sensitivity=all(a<=.9*b for a,b in zip(primary['euclidean_jacobian_errors'],parent['euclidean_jacobian_errors'])),pred_c_values=all(a<=.15 and a<=1.1*b for a,b in zip(primary['per_mode_errors'],base))),scope='Exact conditional readout solve for a linearized gradient surrogate with native amplitudes frozen on1536training sites. Not direct full Jacobian fitting. Same256shared product directions and private3;448opened sites evaluate. Historical chunks do not establish document independence.')
(P/'DOWNSTREAM_READOUT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
