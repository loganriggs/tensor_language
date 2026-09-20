import json
from pathlib import Path
import torch
from quartic_cp import cp_gram
from cp_dictionary import select_atoms
P=Path(__file__).resolve().parent;torch.set_default_dtype(torch.float64);torch.manual_seed(1707);f=[torch.randn(12,6) for _ in range(4)];K=cp_gram(f,f);teacher=torch.randn(3,12);Q=teacher@K;receipts=select_atoms(K,Q,4);chosen=[];errors=[]
for row in receipts:
 gains=[]
 for j in range(12):
  if j in chosen:gains.append(-float('inf'));continue
  ix=chosen+[j];g=K[ix][:,ix];c=torch.linalg.solve(g,Q[:,ix].T).T;gains.append(float(2*(Q[:,ix]*c).sum()-((c@g)*c).sum()))
 assert int(torch.tensor(gains).argmax())==row['selected'][-1];errors.append(abs(max(gains)-row['gain']));chosen=row['selected']
assert max(errors)<1e-8
out=dict(max_absolute_gain_error=max(errors),selections=[r['selected'][-1] for r in receipts],scope='Schur-complement candidate scores choose same atoms as exhaustive exact writer refits on independent small CP dictionary.')
(P/'CP_DICTIONARY_CHECK_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
