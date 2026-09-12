import json
from pathlib import Path
import numpy as np
from real_pair_pencil_v1 import decompose
rng=np.random.default_rng(91871)
a=np.diag([2.,-3.,.2,-.2]);a[2,3]=a[3,2]=1
b=np.diag([1.,-1.,1.,-1.])
u,_,v=np.linalg.svd(rng.normal(size=(4,4)));w=u@np.diag([1.,2.,3.,4.])@v
forms=np.stack([w.T@a@w,w.T@b@w])
_,groups,_,r=decompose(forms)
r['pred_a']=r['reconstruction_error']<=1e-10 and r['eigen_backward_error']<=1e-10 and r['scalar_blocks']==2 and r['pair_blocks']==1
out=Path(__file__).resolve().parent/'REAL_PAIR_PENCIL_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(r);assert r['pred_a']
