"""Finite selective graph edits with an exact coefficient-error budget."""
from pathlib import Path
import torch,json,math
from scipy.optimize import linear_sum_assignment
from audit_profiled_subspaces import gram
from shared_quadratic_products import materialize_mixed
from source_graph_metrics import export,score
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());parent=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']];s=parent['shared_mixed'];root=torch.linalg.inv(d['inverse_root']);L=root@s['left_reader'];R=root@s['right_reader'];W=s['product_weights'].T/d['scales'][:4,None]
n=W.norm(dim=0);balance=(R.norm(dim=0)/L.norm(dim=0)).sqrt();L=L*n.sqrt()*balance;R=R*n.sqrt()/balance;W=W/n
reference=(L,R,W);original=materialize_mixed(*reference);norm2=original.square().sum();atomenergy=gram(reference,reference).diag();cos=W.T@W;used=set();pairs=[]
for flat in torch.argsort(cos.abs().flatten(),descending=True).tolist():
 i,j=divmod(flat,L.shape[1])
 if i<j and i not in used and j not in used:pairs.append((i,j));used.update([i,j])
 if len(used)==L.shape[1]:break

def rotation(pairlist):
 l=L.clone();r=R.clone();w=W.clone()
 for i,j in pairlist:
  if w[:,i]@w[:,j]<0:r[:,j]*=-1;w[:,j]*=-1
  for F in [l,r]:
   a=F[:,i].clone();b=F[:,j].clone();F[:,i]=(a+b)/math.sqrt(2);F[:,j]=(-a+b)/math.sqrt(2)
 return l,r,w

full=rotation(pairs);a=[];b=[];c=[];energies=[]
for i,j in pairs:
 v=full[2][:,i]-full[2][:,j]
 a.extend([full[0][:,i],L[:,i]]);b.extend([full[1][:,i],R[:,i]]);c.extend([v,-v]);energies.append(atomenergy[i]+atomenergy[j])
residual=(torch.stack(a,1),torch.stack(b,1),torch.stack(c,1));N=len(pairs)
K=gram(residual,residual).reshape(N,2,N,2).sum((1,3));energies=torch.stack(energies);order=torch.argsort(K.diag()/energies)
G=K[order[:,None],order];error2=G.cumsum(0).cumsum(1).diag().clamp_min(0)/norm2
fraction=energies[order].cumsum(0)/atomenergy.sum();curve=[dict(pairs=k+1,relative_function_change=float(error2[k].sqrt()),affected_atom_energy_fraction=float(fraction[k])) for k in range(N)]
valid=[k for k in range(N) if error2[k]<=.02**2];count=max(valid)+1 if valid else 0
selected=[pairs[int(i)] for i in order[:count]];new=rotation(selected);hat=materialize_mixed(*new);dense=float((hat-original).norm()/original.norm());implicit=float(error2[count-1].sqrt()) if count else 0.;assert abs(dense-implicit)<1e-8
cross=gram(reference,new)/(atomenergy[:,None]*gram(new,new).diag()[None,:]).sqrt();i,j=linear_sum_assignment(-cross.abs().numpy());matched=cross[i,j].abs()
result=dict(selected_pairs=count,relative_function_change=dense,implicit_replay=abs(dense-implicit),affected_atom_energy_fraction=float(fraction[count-1]) if count else 0.,median_matched_atom_cosine=float(matched.median()),atom_energy_fraction_with_match_below_090=float((atomenergy[i]*(matched<.9)).sum()/atomenergy.sum()),**score(export(*new,d,parent),d))
torch.save(dict(selected_pairs=selected),P/'SELECTIVE_PRODUCT_ROTATION_PAIRS_V1.pt')
out=dict(result=result,curve=curve,predictions=dict(pred_a_replay=abs(dense-implicit)<1e-8,pred_b_substantial_freedom=result['affected_atom_energy_fraction']>=.5),scope='Selected using coefficient error only. Energy is sum of pre-cancellation atom squared norms, not causal importance. Unchanged256shared+256private products; no cheaper or semantic circuit claim. Largest prefix under a fixed2%metric budget, not globally optimal matching or rewrite.')
(P/'SELECTIVE_PRODUCT_ROTATIONS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(result,indent=2));print(out['predictions'])
