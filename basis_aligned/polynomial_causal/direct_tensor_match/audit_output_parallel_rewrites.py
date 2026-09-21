"""Finite product rotations test approximate non-identification from shared outputs."""
from pathlib import Path
import torch,json,math
from scipy.optimize import linear_sum_assignment
from shared_quadratic_products import materialize_mixed
from source_graph_metrics import export,score
P=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);meta=json.loads((P/'PROFILED_PARTIAL_GRAPH_FIT_V1.json').read_text());p=torch.load(P/'PROFILED_PARTIAL_GRAPH_PROGRAMS_V1.pt',weights_only=True)[meta['winner']];root=torch.linalg.inv(d['inverse_root']);s=p['shared_mixed'];L=root@s['left_reader'];R=root@s['right_reader'];W=s['product_weights'].T/d['scales'][:4,None]
# Unit output directions and balanced input-factor norms preserve every atom.
n=W.norm(dim=0);balance=(R.norm(dim=0)/L.norm(dim=0)).sqrt();L=L*n.sqrt()*balance;R=R*n.sqrt()/balance;W=W/n
original=materialize_mixed(L,R,W);cos=W.T@W;N=L.shape[1]
def rotate(L,R,W,pairs):
 L=L.clone();R=R.clone();W=W.clone()
 for i,j in pairs:
  if W[:,i]@W[:,j]<0:W[:,j]*=-1;R[:,j]*=-1
  for F in [L,R]:
   a=F[:,i].clone();b=F[:,j].clone();F[:,i]=(a+b)/math.sqrt(2);F[:,j]=(-a+b)/math.sqrt(2)
 return L,R,W
used=set();pairs=[]
for flat in torch.argsort(cos.abs().flatten(),descending=True).tolist():
 i,j=divmod(flat,N)
 if i<j and i not in used and j not in used:pairs.append((i,j));used.update([i,j])
 if len(used)==N:break
permutation=torch.randperm(N,generator=torch.Generator().manual_seed(806)).tolist();randompairs=list(zip(permutation[::2],permutation[1::2]))
def gram(X,Y):
 l,r,w=X;a,b,v=Y
 return (w.T@v)*.5*((l.T@a)*(r.T@b)+(l.T@b)*(r.T@a))
def matched(X,Y):
 K=gram(X,Y);c=K/(gram(X,X).diag()[:,None]*gram(Y,Y).diag()[None,:]).sqrt();i,j=linear_sum_assignment(-c.abs().numpy());return float(c[i,j].abs().median())
# Exactly equal output factors give an exact finite rotation, independently of inputs.
g=torch.Generator().manual_seed(807);a=torch.randn(8,2,generator=g,dtype=torch.double);b=torch.randn(8,2,generator=g,dtype=torch.double);w=torch.randn(4,1,generator=g,dtype=torch.double).repeat(1,2)
control=float((materialize_mixed(*rotate(a,b,w,[(0,1)]))-materialize_mixed(a,b,w)).norm()/materialize_mixed(a,b,w).norm());assert control<1e-10
pair_diagnostics=[]
for i,j in pairs:
 sign=1 if float(W[:,i]@W[:,j])>=0 else -1
 F=torch.stack([L[:,i],R[:,i],L[:,j],sign*R[:,j]],1);G=F.T@F
 S=torch.zeros(4,4,dtype=L.dtype)
 for a,b,value in [(0,1,-.25),(2,3,.25),(0,3,.25),(2,1,.25)]:S[a,b]=value;S[b,a]=value
 error2=(W[:,i]-sign*W[:,j]).square().sum()*torch.trace(S@G@S@G)
 pair_diagnostics.append(dict(i=i,j=j,output_abs_cosine=abs(float(cos[i,j])),individual_squared_change_over_function_energy=float(error2/original.square().sum())))
records=[]
for name,pairlist in [('nearest_output',pairs),('random_pairs',randompairs)]:
 new=rotate(L,R,W,pairlist);Q=materialize_mixed(*new);error=float((Q-original).norm()/original.norm());rec=dict(method=name,median_pair_output_abs_cosine=float(torch.tensor([abs(float(cos[i,j])) for i,j in pairlist]).median()),relative_function_change=error,median_matched_atom_abs_cosine=matched((L,R,W),new),**score(export(*new,d,p),d));records.append(rec);print(rec)
out=dict(records=records,largest_pair_changes=sorted(pair_diagnostics,key=lambda r:r['individual_squared_change_over_function_energy'],reverse=True)[:10],sum_individual_pair_squared_change=sum(r['individual_squared_change_over_function_energy'] for r in pair_diagnostics),equal_output_exact_replay=control,predictions=dict(pred_a_exact_control=control<1e-10,pred_b_approximate_freedom=records[0]['relative_function_change']<=.02 and records[0]['median_matched_atom_abs_cosine']<=.9),scope='Finite rotation at unchanged256shared products and unchanged nominal coefficient storage. Global polynomial change measured in existing covariance coefficient metric. Does not prove exact nonuniqueness of the trained tensor, semantic equivalence or causal adoption.')
(P/'OUTPUT_PARALLEL_REWRITES_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out['predictions'])
