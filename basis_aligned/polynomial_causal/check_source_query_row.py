"""Independent dense-attention oracle for the exact single-token row+column fold."""
from pathlib import Path
import torch,json
from fixed_query_source_column import source_column
from source_query_row import query_row

torch.manual_seed(2301);torch.set_num_threads(2);records=[]
for t in (5,11):
 b,h,e,d=3,3,4,12;dt=torch.float64;qs=[torch.randn(b,t,h,e,dtype=dt) for _ in range(2)];ks=[torch.randn(b,t,h,e,dtype=dt) for _ in range(2)];v=torch.randn(b,t,h,e,dtype=dt);w=torch.randn(d,d,dtype=dt);pos=torch.tensor([0,t//2,t-1]);batch=torch.arange(b)
 newq=[a.clone() for a in qs];newk=[a.clone() for a in ks];newv=v.clone()
 for a in newq+newk+[newv]:a[batch,pos]=torch.randn(b,h,e,dtype=dt)
 def dense(q,k,value):
  scores=[torch.einsum('bthe,bshe->bhts',a,c)/e for a,c in zip(q,k)];pat=(scores[0]*scores[1]).tril()
  return torch.einsum('bhts,bshe->bthe',pat,value).reshape(b,t,d)@w.T
 col=source_column(*qs,ks[0][batch,pos],ks[1][batch,pos],v[batch,pos],newk[0][batch,pos],newk[1][batch,pos],newv[batch,pos],w,pos)
 row=query_row(qs[0][batch,pos],qs[1][batch,pos],newq[0][batch,pos],newq[1][batch,pos],*newk,newv,w,pos)
 pred=col.clone();pred[batch,pos]+=row;actual=dense(newq,newk,newv)-dense(qs,ks,v)
 relative=float((pred-actual).norm()/actual.norm());wrong=query_row(qs[0][batch,pos],qs[1][batch,pos],newq[0][batch,pos],newq[1][batch,pos],*ks,v,w,pos)
 mixed=float((row-wrong).norm()/actual.norm());assert relative<1e-12 and mixed>.05
 records.append(dict(tokens=t,relative_error=relative,query_source_mixed_live=mixed))
p=Path(__file__).with_name('SOURCE_QUERY_ROW_CPU_V1.json');assert not p.exists();p.write_text(json.dumps(dict(records=records,scope='Exact independent-port single-token attention fold; native state adapter and installed validation still pending'),indent=2)+'\n');print(records)
