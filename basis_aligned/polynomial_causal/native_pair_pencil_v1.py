"""Two fixed MLP17 output forms, CPU-only generic real pencil screen.

A eigen/reconstruction <=1e-8; B basis condition <=1e4;
C greedy32-reader block error <= independent16+16 spectral error.
No text, no producer compression, no semantic/circuit claim.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from real_pair_pencil_v1 import decompose,truncate

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter()
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()
assert json.loads((P/'REAL_PAIR_PENCIL_V1_CONTROL.json').read_text())['pred_a']
binding=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_BINDING.json').read_text())['files']
ck=next(k for k in binding if k.endswith('/pytorch_model.bin'));assert digest(ck)==binding[ck]
source=P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt';receipt=json.loads((P/'COUPLED_QUARTIC_LBFGS_V1_RESULT.json').read_text());assert digest(source)==receipt['artifact_sha256']
program=torch.load(source,weights_only=True);writer=program['output_writers']
state=torch.load(ck,weights_only=True,mmap=True);u=state['lm_head.weight'].double();mean=u.mean(0)
readers=u.T@(u@writer)-len(u)*mean[:,None]*(mean@writer)[None,:];del u
metric_error=float((writer.T@readers-torch.eye(2)).abs().max());assert metric_error<=1e-8
l,r,d=[state[f'transformer.h.17.mlp.{name}.weight'].double() for name in ('Left','Right','Down')]
output_coefficients=readers.T@d
forms=[]
for row in output_coefficients:
    raw=l.T@(row[:,None]*r);forms.append((raw+raw.T)/2)
forms=torch.stack(forms).numpy()
dual,groups,blocks,diag=decompose(forms)
chosen,error,count=truncate(forms,dual,groups,blocks,32)
baseline=np.zeros_like(forms)
for m,form in enumerate(forms):
    eigen,vectors=np.linalg.eigh(form);ids=np.argsort(abs(eigen))[-16:]
    baseline[m]=(vectors[:,ids]*eigen[ids])@vectors[:,ids].T
baseline_error=float(np.linalg.norm(baseline-forms)/np.linalg.norm(forms))
artifact=P/'NATIVE_PAIR_PENCIL_V1_PROGRAM.pt';assert not artifact.exists()
torch.save(dict(dual_readers=torch.from_numpy(dual),groups=groups,cores=[torch.from_numpy(z) for z in blocks],selected_blocks=chosen,output_writers=writer,source_sha256=digest(source)),artifact)
result=dict(pred_a=max(diag['reconstruction_error'],diag['eigen_backward_error'])<=1e-8,pred_b=diag['frame_condition']<=1e4,pred_c=error<=baseline_error,diagnostics=diag,writer_metric_error=metric_error,selected_readers=count,selected_blocks=len(chosen),block_truncation_error=error,independent_spectral_error=baseline_error,full_outer_floats=int(dual.size+sum(z.size for z in blocks)+writer.numel()),truncated_outer_floats=int(count*1152+sum(blocks[i].size for i in chosen)+writer.numel()),artifact_sha256=digest(artifact),checkpoint_sha256=binding[ck],seconds=time.perf_counter()-start,scope='Exact pair representation screen in MLP17 input coordinates, before folding dual readers through full native MLP16. Outer-form Frobenius error is not composed quartic error or native effect fidelity. Generic pair blocks do not imply semantic independence; truncation is a heuristic.')
out=P/'NATIVE_PAIR_PENCIL_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
