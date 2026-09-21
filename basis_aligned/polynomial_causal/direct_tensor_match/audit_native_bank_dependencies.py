"""Finite polynomial-evaluation rank tests for native-derived quadratic banks.
Full column rank witnesses absence of exact root identities up to numerical error.
Not a global norm conditioning certificate or a native behavioral evaluation.
"""
import json,hashlib,itertools
from pathlib import Path
import torch
P=Path(__file__).resolve().parent

def design(Q,seed):
 k,d,_=Q.shape;pairs=list(itertools.combinations_with_replacement(range(k),2));gen=torch.Generator().manual_seed(seed);x=torch.randn(max(128,8*len(pairs)),d,generator=gen,dtype=Q.dtype);reads=torch.stack([(x@q*x).sum(-1) for q in Q],1);return torch.stack([reads[:,i]*reads[:,j]*(1 if i==j else 2**.5) for i,j in pairs],1)

def assess(Q):
 rows=[]
 for seed in (933,934):
  F=design(Q,seed);norm=F.norm(dim=0);assert bool((norm>0).all());X=F/norm;sv=torch.linalg.svdvals(X);rank=int((sv>1e-10*sv[0]).sum());rows.append(dict(seed=seed,rows=len(F),columns=F.shape[1],rank=rank,min_relative_singular=float(sv[-1]/sv[0]),singular_values=sv.tolist()))
 return rows

def main():
 torch.set_num_threads(2);torch.set_grad_enabled(False);dtype=torch.float64
 # Independent coordinate squares, then exact duplicate and linear-combination controls.
 Q=torch.diag_embed(torch.eye(3,dtype=dtype));controls=[]
 for name,q,expected in [('independent',Q,6),('duplicate',torch.cat((Q,Q[:1])),6),('linear_combination',torch.cat((Q,(Q[0]+Q[1])[None])),6)]:
  rows=assess(q);assert all(r['rank']==expected for r in rows);controls.append(dict(name=name,expected_rank=expected,rows=rows))
 sources=[('earlier_four_feature_bank','QUARTIC_BANK_CORE_V1.pt','core'),('six_product_four_feature_bank','ROOT_ARCHIVE_METRIC_V1.pt','Q')];rows=[];hashes={}
 for name,file,key in sources:
  hashes[file]=hashlib.sha256((P/file).read_bytes()).hexdigest();q=torch.load(P/file,weights_only=True)[key].double();rows.append(dict(name=name,shape=list(q.shape),evaluation=assess(q)))
 file='SHARED_PRODUCT_NATIVE_INPUTS_V1.pt';hashes[file]=hashlib.sha256((P/file).read_bytes()).hexdigest();d=torch.load(P/file,weights_only=True);q=torch.stack([Q.double() for p in d['pairs'] for Q in p['Qs']]);rows.append(dict(name='six_original_source_reads',shape=list(q.shape),evaluation=assess(q)))
 out=dict(controls=controls,hashes=hashes,banks=rows,predictions=dict(any_bank_has_root_identity=any(r['rank']<r['columns'] for b in rows for r in b['evaluation'])),scope='No fitting. The first two banks are exact input-span representations of learned full-input replacement features, not arbitrary amplitude slices or the entire trained teacher. Third bank is six original source quadratic reads at1152inputwidth. Finite full-rank evaluation rules out exact root dependencies numerically within each frozen bank only. It does not exclude new features, near-identities in other metrics, or a different graph.')
 (P/'NATIVE_BANK_DEPENDENCIES_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(banks=rows,predictions=out['predictions']),indent=2))
if __name__=='__main__':main()
