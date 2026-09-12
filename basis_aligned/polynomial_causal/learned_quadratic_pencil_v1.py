"""Fixed learned mixed core -> common quadratic basis -> rank16 readers.

A exact native write replay <=1e-8; B frame condition<=1e4;
C rank16 truncation reduces full mixed native error by >=20%.
CPU, frozen coefficients, no native-data fitting or selection.
"""
from pathlib import Path
import json,time
import numpy as np
import torch
from real_pair_pencil_v1 import decompose
from quadratic_product_core_v1 import quadratic_values,features
from sparse_path_stability_atlas_v1 import digest

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
start=time.perf_counter()
source=P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_PROGRAMS.pt'
receipt=json.loads((P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_RESULT.json').read_text())
assert digest(source)==receipt['artifact_sha256'] and receipt['pred_a']
p=torch.load(source,weights_only=True)['programs'][0]
b,n=p['input_readers'],p['inner_weights'];mix=p['mixing'];ij=p['pairs']
forms=torch.zeros(2,len(b),len(b))
for e,(i,j) in enumerate(ij.T):
    forms[:,i,j]=mix[e] if i==j else mix[e]/2
    forms[:,j,i]=forms[:,i,j]
dual,groups,blocks,diagnostic=decompose(forms.numpy())
dual=torch.from_numpy(dual)
blockforms=torch.zeros_like(forms)
for ids,core in zip(groups,blocks):
    for m in range(2):
        for a,i in enumerate(ids):
            for c,j in enumerate(ids):blockforms[m,i,j]=float(core[m,a,c])
x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double()
pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double()
den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0]
writer=p['output_writers']
baseline=features(b,n,x)@mix@writer.T/den[:,None]
q=quadratic_values(b,n,x)@dual.T
def write(values):
    return torch.einsum('ni,mij,nj->nm',values,blockforms,values)@writer.T/den[:,None]
exact=write(q)
replay=float((exact-baseline).norm()/baseline.norm())
baseerror=float((baseline-reference).norm()/reference.norm())
original=torch.einsum('kdi,ki,kei->kde',b,n,b)
newb=[];newn=[];errors=[]
for row in dual:
    matrix=torch.einsum('k,kij->ij',row,original)
    values,vectors=torch.linalg.eigh(matrix)
    selected=values.abs().argsort()[-16:]
    newb.append(vectors[:,selected]);newn.append(values[selected])
    errors.append(float((values.square().sum()-values[selected].square().sum()).clamp_min(0).sqrt()/values.norm()))
newb=torch.stack(newb);newn=torch.stack(newn)
approx=write(quadratic_values(newb,newn,x))
error=float((approx-reference).norm()/reference.norm())
artifact=P/'LEARNED_QUADRATIC_PENCIL_V1_PROGRAM.pt'
out=P/'LEARNED_QUADRATIC_PENCIL_V1_RESULT.json'
assert not artifact.exists() and not out.exists()
torch.save(dict(input_readers=newb,inner_weights=newn,groups=groups,
                cores=[torch.from_numpy(z) for z in blocks],output_writers=writer,
                source_sha256=digest(source)),artifact)
result=dict(pred_a=max(replay,diagnostic['reconstruction_error'],abs(baseerror-receipt['reports'][0]['coupled_native_error']))<=1e-8,
            pred_b=diagnostic['frame_condition']<=1e4,pred_c=error<=.8*baseerror,
            diagnostics=diagnostic,exact_write_replay_error=replay,
            full_mixed_native_error=baseerror,truncated_native_error=error,
            transformed_quadratic_relative_errors=errors,
            fitted_floats=newb.numel()+newn.numel()+writer.numel()+sum(z.size for z in blocks),
            scalar_products=sum(len(g)*(len(g)+1)//2 for g in groups),
            artifact_sha256=digest(artifact),seconds=time.perf_counter()-start,
            scope='CPU frozen learned32-quadratic core reencoding; differs from native1152-outer pencil. Rank16 truncation without output refit. Native cache diagnoses only; no target coefficient approximation or circuit identification claim.')
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
assert result['pred_a']
