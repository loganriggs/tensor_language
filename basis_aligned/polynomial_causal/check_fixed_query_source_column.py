from pathlib import Path
import torch,json
from fixed_query_source_column import source_column,price

torch.set_num_threads(2);torch.manual_seed(2201);records=[]
for t in (5,11):
 b,h,e,d=3,3,4,12;dt=torch.float64
 q1,q2,k1,k2,v=[torch.randn(b,t,h,e,dtype=dt) for _ in range(5)]
 w=torch.randn(d,h*e,dtype=dt);positions=torch.tensor([0,t//2,t-1]);batch=torch.arange(b)
 k11,k21,v1=[torch.randn(b,h,e,dtype=dt) for _ in range(3)]
 def direct(a,c,value):
  p=(torch.einsum('bthe,bshe->bhts',q1,a)*torch.einsum('bthe,bshe->bhts',q2,c)/e**2).tril()
  return torch.einsum('bhts,bshe->bthe',p,value).reshape(b,t,d)@w.T
 ke1,ke2,ve=k1.clone(),k2.clone(),v.clone();ke1[batch,positions]=k11;ke2[batch,positions]=k21;ve[batch,positions]=v1
 actual=source_column(q1,q2,k1[batch,positions],k2[batch,positions],v[batch,positions],k11,k21,v1,w,positions)
 expected=direct(ke1,ke2,ve)-direct(k1,k2,v)
 cache=torch.stack([v[batch,positions][:,j]@w[:,j*e:(j+1)*e].T for j in range(h)],1)
 cached=source_column(q1,q2,k1[batch,positions],k2[batch,positions],v[batch,positions],k11,k21,v1,w,positions,cache)
 assert torch.equal(cached,actual)
 rel=float((actual-expected).norm()/expected.norm());abs_err=float((actual-expected).abs().max())
 causal=float(actual[torch.arange(t)[None]<positions[:,None]].abs().max())
 wrong=source_column(q1,q2,k1[batch,positions],k2[batch,positions],v[batch,positions],k11,k2[batch,positions],v1,w,positions)
 live=float((wrong-expected).norm()/expected.norm());assert rel<1e-12 and causal==0 and live>.1
 records.append(dict(tokens=t,relative_error=rel,max_absolute_error=abs_err,pre_source_write=causal,omit_second_key_relative_error=live))
p=Path(__file__).with_name('FIXED_QUERY_SOURCE_COLUMN_CPU_V2.json');assert not p.exists();p.write_text(json.dumps(dict(records=records,prices=[price(t) for t in (12,20,32)],scope='Conditional exact algebra, synthetic ports; native installed replay still required'),indent=2)+'\n');print(records)
