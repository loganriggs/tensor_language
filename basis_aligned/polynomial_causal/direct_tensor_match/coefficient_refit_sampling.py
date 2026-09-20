import itertools,json,math,time
from pathlib import Path
import torch
from quadratic_product_gram import product_gram
P=Path(__file__).resolve().parent

def features(Q,R,ix):
 i,j,k,l=ix.T
 return ((Q[:,i,j]*R[:,k,l]+R[:,i,j]*Q[:,k,l]+Q[:,i,k]*R[:,j,l]+R[:,i,k]*Q[:,j,l]+Q[:,i,l]*R[:,j,k]+R[:,i,l]*Q[:,j,k])/6).T

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(1744);d=8;idx=torch.tensor(list(itertools.product(range(d),repeat=4)));out=P/'COEFFICIENT_REFIT_SAMPLING_V1.json';assert not out.exists();records=[];baselines=[];checks=[];start=time.perf_counter()
 for family in ['dense','diagonal','shared_quadratic']:
  Q=torch.randn(12,d,d);R=torch.randn(12,d,d);Q=(Q+Q.transpose(-1,-2))/2;R=(R+R.transpose(-1,-2))/2
  if family=='diagonal':Q=torch.diag_embed(Q.diagonal(dim1=-2,dim2=-1));R=torch.diag_embed(R.diagonal(dim1=-2,dim2=-1))
  if family=='shared_quadratic':Q=Q[:1].expand(12,-1,-1).clone()
  C=torch.randn(3,12);F=features(Q,R,idx);Y=F@C.T;norm=Y.square().sum();G=product_gram(Q,R,Q,R);check=float((G-F.T@F).norm()/G.norm());assert check<1e-11;checks.append(dict(family=family,gram_replay=check))
  for width in [2,4,8]:
   X=F[:,:width];exact=torch.linalg.lstsq(X,Y,driver='gelsd',rcond=1e-12).solution;exact_e=float((X@exact-Y).square().sum()/norm);gain=1-exact_e;baselines.append(dict(family=family,width=width,exact_error=math.sqrt(max(0,exact_e)),full_design_condition=float(torch.linalg.cond(X))))
   for n,seed in itertools.product([16,64,256,1024,4096],range(12)):
    gen=torch.Generator();gen.manual_seed(10000+n*20+seed);ix=torch.randint(len(X),(n,),generator=gen);Xs=X[ix];Ys=Y[ix];solve=torch.linalg.lstsq(Xs,Ys,driver='gelsd',rcond=1e-12);c=solve.solution;e=float((X@c-Y).square().sum()/norm);train_den=float(Ys.square().sum());train_error=None if train_den==0 else float((Xs@c-Ys).norm()/Ys.norm());rank=int(solve.rank);condition=float(solve.singular_values[0]/solve.singular_values[-1]) if rank==width else None;records.append(dict(family=family,width=width,queries=n,seed=seed,training_error=train_error,full_error=math.sqrt(max(0,e)),optimal_gain_fraction=(1-e)/gain,design_rank=rank,design_condition=condition))
  print(family,'done',flush=True)
 out.write_text(json.dumps(dict(records=records,baselines=baselines,checks=checks,seconds=time.perf_counter()-start,scope='540 finite coefficient writerrefits vs exact enumeratedoptimum; retained rootset fixed. Rank deficient designs visible, no optimizer stochasticity. Synthetic sampling control, not native samplecomplexity certificate.'),indent=2)+'\n')
 print('fits',len(records),'seconds',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
