"""Executable product graph with dictionaries shared by pairs of consumers."""
import torch
from quadratic_pair_blocks import products
from overlap_varpro import OverlapMetric
GROUPS=((0,1),(0,2),(1,2))

def from_common(program):
 A,B=program['input_basis'].chunk(2,1);assert A.shape==B.shape
 bases={'0':A.clone(),'1':B.clone(),'2':((A+B)/2**.5).clone()};pairs={}
 for j in range(3):
  p=dict(program['pairs'][str(j)]);a,b=p['shared_map'].chunk(2,0)
  p['shared_map']=(torch.cat([a,b]) if j==0 else torch.cat([a-b,2**.5*b]) if j==1 else torch.cat([b-a,2**.5*a])).clone();pairs[str(j)]=p
 return dict(input_bases=bases,pairs=pairs)

def expand(program):
 out={}
 for j,(a,b) in enumerate(GROUPS):
  p=program['pairs'][str(j)];basis=torch.cat([program['input_bases'][str(a)],program['input_bases'][str(b)]],1)
  n=len(p['shared_indices'])+len(p['private_indices']);reader=basis.new_zeros(basis.shape[0],n);reader[:,p['shared_indices']]=basis@p['shared_map'];reader[:,p['private_indices']]=p['private_reader'];q={k:v for k,v in p.items() if k not in ('shared_indices','private_indices','shared_map','private_reader')};q['shared_reader']=reader;out[str(j)]=q
 return out

def source_reads(z,program):
 shared={k:z@basis for k,basis in program['input_bases'].items()};out=[]
 for j,(a,b) in enumerate(GROUPS):
  p=program['pairs'][str(j)];n=len(p['shared_indices'])+len(p['private_indices']);t=z.new_zeros(*z.shape[:-1],n)
  t[...,p['shared_indices']]=torch.cat([shared[str(a)],shared[str(b)]],-1)@p['shared_map'];t[...,p['private_indices']]=z@p['private_reader']
  q=products(t,p['product_indices'])@p['product_weights'];q+=torch.stack([z@p[k+'_linear']+p[k+'_bias'] for k in ('a','b')],-1);out.append(q)
 return torch.cat(out,-1)

def price(program):
 from pack_reader_graph_artifacts import counts
 matrices=list(program['input_bases'].values())+[p[k] for p in program['pairs'].values() for k in ('shared_map','private_reader')]
 projection=sum(a.numel() for a in matrices);nonlinear=sum(len(p['product_weights']) for p in program['pairs'].values());readout=sum(p['product_weights'].numel() for p in program['pairs'].values())
 integers=sum(p[k].numel() for p in program['pairs'].values() for k in ('shared_indices','private_indices','product_indices'))
 c=counts(program)
 return dict(stored_floats=c['logical_float_coefficients'],physical_storage_floats=c['backing_storage_floats'],index_integers=integers,projection_multiplications=projection,activation_products=nonlinear,source_total_multiplications=projection+nonlinear+readout)

class PairwiseOverlapMetric(OverlapMetric):
 def reader(self,params,j):
  a,b=GROUPS[j];basis=torch.cat([params[a],params[b]],1);maps,private=params[3+2*j:5+2*j];t=self.templates[j];n=len(t['shared_indices'])+len(t['private_indices'])
  return basis.new_zeros(basis.shape[0],n).index_copy(1,t['shared_indices'],basis@maps).index_copy(1,t['private_indices'],private)

def parameters_from_program(program,transform):
 params=[transform@program['input_bases'][str(j)] for j in range(3)]
 norms=[p.norm(dim=0).clamp_min(1e-20) for p in params];params=[p/n for p,n in zip(params,norms)]
 for j,(a,b) in enumerate(GROUPS):
  p=program['pairs'][str(j)];params.extend([torch.cat([norms[a],norms[b]])[:,None]*p['shared_map'],transform@p['private_reader']])
 return params
