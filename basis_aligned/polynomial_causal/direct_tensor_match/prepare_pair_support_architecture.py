"""Freeze native MLP16 factors for architecture-preserving coefficient nulls."""
from pathlib import Path
import json,torch,hashlib,time
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
ck=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();D=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'][0].double();C=torch.stack([lam*(D.T@pair[k]) for pair in d['pairs'] for k in ('a','b')]);Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']]);replays=[]
for c,q in zip(C,Q):
 raw=L.T@(c[:,None]*R);hat=(raw+raw.T)/2;replays.append(float((hat-q).norm()/q.norm()))
assert max(replays)<1e-10
artifact=dict(L=L,R=R,C=C,root=torch.linalg.inv(d['inverse_root']));torch.save(artifact,P/'PAIR_SUPPORT_ARCHITECTURE_INPUTS_V1.pt')
meta=dict(checkpoint=str(ck),factor_shapes={k:list(v.shape) for k,v in artifact.items()},native_coefficient_replays=replays,sha256=hashlib.sha256((P/'PAIR_SUPPORT_ARCHITECTURE_INPUTS_V1.pt').read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Same six folded MLP16 source forms. lam is the h17 residual coefficient already present in cached targets. Native L/R held fixed; only six channel-coefficient rows are randomized in planned nulls.')
(P/'PAIR_SUPPORT_ARCHITECTURE_PREFLIGHT_V1.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
