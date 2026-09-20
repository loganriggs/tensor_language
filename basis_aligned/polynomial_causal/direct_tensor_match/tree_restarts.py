"""Discriminate a matched-rank optimization trap from a representation limit."""
import json,itertools,time
from pathlib import Path
import torch
from sweep import target_cases,fit
p=Path(__file__).resolve().parent;out=p/'TREE_RESTARTS_V1.json';assert not out.exists();torch.set_num_threads(1)
c=next(x for x in target_cases() if x['name']=='quartic_tree');rows=[];start=time.perf_counter()
for opt,lr,seed in itertools.product(['adam','muon'],[.005,.05,.2],range(8)):
 r,_=fit(c['target'],6,4,'tree',2,opt,lr,seed,10000);rows.append(r)
 out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Same planted rank2tree as initialtoy,48longer randomrestarts, no width expansion or activationtraining'),indent=2)+'\n')
 print(json.dumps({k:r[k] for k in ['optimizer','lr','seed','relative_error','steps']}),flush=True)
