"""Overlap selected reader blocks across pair banks, preserving private bypasses."""
import torch
from quadratic_pair_blocks import products
from shared_private_metric import gram

def product_factors(reader,indices):
 i,j,kind=indices.long();u=reader[:,i];v=reader[:,j]
 return torch.where((kind==1)[None],u+v,u),torch.where((kind==1)[None],u-v,v)

def decode(pair):
 L,R=product_factors(pair['shared_reader'],pair['product_indices']);raw=torch.einsum('ir,ro,jr->oij',L,pair['product_weights'],R)
 return (raw+raw.transpose(-1,-2))/2

def select_blocks(pair,basis,transform,count):
 """Knapsack on individual block perturbations, not a global optimal edit claim."""
 reader=pair['shared_reader'];projected=basis@(basis.T@reader);L,R=product_factors(reader,pair['product_indices']);l,r=product_factors(projected,pair['product_indices']);L,R,l,r=[transform@x for x in (L,R,l,r)]
 i,j,kind=pair['product_indices'].tolist();groups={}
 for row,(a,b) in enumerate(zip(i,j)):groups.setdefault((a,b),[]).append(row)
 blocks=[]
 for ids,rows in groups.items():
  columns=sorted(set(ids));A,B,C,D=[x[:,rows] for x in (L,R,l,r)];G=gram(A,B,A,B)+gram(C,D,C,D)-gram(A,B,C,D)-gram(C,D,A,B);W=pair['product_weights'][rows];score=float((W*(G@W)).sum().clamp_min(0));blocks.append((columns,score))
 # Exact cardinality minimization of this separable heuristic score.
 dp={0:(0.,[])}
 for index,(columns,cost) in enumerate(blocks):
  updated=dict(dp)
  for n,(value,selected) in dp.items():
   new=n+len(columns)
   if new<=count and (new not in updated or value+cost<updated[new][0]):updated[new]=(value+cost,selected+[index])
  dp=updated
 if count not in dp:raise ValueError('Requested column count incompatible with complete pencil blocks')
 chosen=sorted(c for b in dp[count][1] for c in blocks[b][0]);assert len(set(chosen))==count
 return torch.tensor(chosen,dtype=torch.int64),dict(separable_block_cost=dp[count][0],selected_blocks=len(dp[count][1]))

def factor_bundle(bundle,basis,selections):
 pairs={}
 for key,p in bundle.items():
  selected=selections[key];mask=torch.ones(p['shared_reader'].shape[1],dtype=torch.bool);mask[selected]=False;private=torch.where(mask)[0]
  q={k:v for k,v in p.items() if k!='shared_reader'}
  q.update(shared_indices=selected,private_indices=private,shared_map=basis.T@p['shared_reader'][:,selected],private_reader=p['shared_reader'][:,private].clone());pairs[key]=q
 return dict(input_basis=basis,pairs=pairs)

def expand(program):
 basis=program['input_basis'];out={}
 for key,p in program['pairs'].items():
  n=len(p['shared_indices'])+len(p['private_indices']);reader=basis.new_empty(basis.shape[0],n);reader[:,p['shared_indices']]=basis@p['shared_map'];reader[:,p['private_indices']]=p['private_reader']
  q={k:v for k,v in p.items() if k not in ('shared_indices','private_indices','shared_map','private_reader')};q['shared_reader']=reader;out[key]=q
 return out

def source_reads(z,program):
 s=z@program['input_basis'];out=[]
 for key in sorted(program['pairs'],key=int):
  p=program['pairs'][key];t=z.new_empty(*z.shape[:-1],len(p['shared_indices'])+len(p['private_indices']));t[...,p['shared_indices']]=s@p['shared_map'];t[...,p['private_indices']]=z@p['private_reader']
  q=products(t,p['product_indices'])@p['product_weights'];q+=torch.stack([z@p[k+'_linear']+p[k+'_bias'] for k in ('a','b')],-1);out.append(q)
 return torch.cat(out,-1)

def component_scalars(z,h,program):
 q=source_reads(z,program);scale=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();out=[]
 for j in range(3):
  p=program['pairs'][str(j)];out.append(((h@p['h_reader']-.5*q[:,2*j])/scale-p['alpha'])*(q[:,2*j+1]/scale-p['beta']))
 return torch.stack(out,1)

def price(program):
 unique={};indices=0
 def visit(x):
  nonlocal indices
  if isinstance(x,dict):
   for v in x.values():visit(v)
  elif torch.is_tensor(x):
   if x.is_floating_point():unique[x.data_ptr()]=x
   else:indices+=x.numel()
 visit(program);matrices=[program['input_basis']]+[p[k] for p in program['pairs'].values() for k in ('shared_map','private_reader')];projection=sum(a.numel() for a in matrices);products_n=sum(len(p['product_weights']) for p in program['pairs'].values());readout=sum(p['product_weights'].numel() for p in program['pairs'].values());block_add=sum(2*int((p['product_indices'][2]==1).sum()) for p in program['pairs'].values())
 return dict(stored_floats=sum(a.numel() for a in unique.values()),index_integers=indices,activation_products=products_n,projection_multiplications=projection,source_total_multiplications=projection+products_n+readout,source_additions=sum(a.shape[1]*max(0,a.shape[0]-1) for a in matrices)+sum(2*(len(p['product_weights'])-1) for p in program['pairs'].values())+block_add)

def refit_pair(pair,target,transform,ridge=1e-12):
 L,R=product_factors(pair['shared_reader'],pair['product_indices']);L,R=transform@L,transform@R;norm=L.norm(dim=0)*R.norm(dim=0);L=L/L.norm(dim=0);R=R/R.norm(dim=0);T=transform@target@transform
 G=gram(L,R,L,R);rhs=torch.einsum('ir,oij,jr->ro',L,T,R);A=G+ridge*torch.eye(len(G),dtype=G.dtype);W=torch.linalg.solve(A,rhs)
 residual=float((A@W-rhs).norm()/rhs.norm());old=pair['product_weights']*norm[:,None]
 def objective(w):return float(((w*(A@w)).sum()-2*(rhs*w).sum()+T.square().sum())/T.square().sum())
 old_objective=objective(old);new_objective=objective(W);assert residual<1e-8 and new_objective<=old_objective+1e-8
 return W/norm[:,None],dict(normal_equation_residual=residual,old_regularized_objective=old_objective,new_regularized_objective=new_objective,product_gram_condition=float(torch.linalg.cond(A)))
