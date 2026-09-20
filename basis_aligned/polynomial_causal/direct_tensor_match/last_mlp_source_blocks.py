"""Exact source-block expansion of a bilinear MLP numerator."""
import torch

def terms(L,R,D,sources):
 left={k:x@L.T for k,x in sources.items()};right={k:x@R.T for k,x in sources.items()};names=list(sources);out={}
 for i,a in enumerate(names):
  for b in names[i:]:
   products=left[a]*right[b]
   if a!=b:products=products+left[b]*right[a]
   out[a+'_'+b]=products@D.T
 return out

def toy_check():
 gen=torch.Generator().manual_seed(261025);rand=lambda *shape:torch.randn(*shape,generator=gen,dtype=torch.float64)
 L,R,D=rand(9,7),rand(9,7),rand(5,9);sources={k:rand(13,7) for k in ['r','m','a']};t=terms(L,R,D,sources);h=sum(sources.values());target=((h@L.T)*(h@R.T))@D.T;error=float((sum(t.values())-target).norm()/target.norm());assert error<1e-12
 independent=((sources['m']@L.T)*(sources['m']@R.T))@D.T;selferror=float((t['m_m']-independent).norm()/independent.norm());assert selferror==0
 # Removing all terms touching m equals evaluating the same polynomial on r+a.
 without=sum(v for k,v in t.items() if 'm' not in k.split('_'));z=sources['r']+sources['a'];reference=((z@L.T)*(z@R.T))@D.T;marginal=float((without-reference).norm()/reference.norm());assert marginal<1e-12
 return dict(sum_replay=error,self_replay=selferror,source_union_replay=marginal)
if __name__=='__main__':
 import json
 from pathlib import Path
 r=toy_check();Path(__file__).with_name('LAST_MLP_SOURCE_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
