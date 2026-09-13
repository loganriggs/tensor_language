#!/usr/bin/env python3
# BQGATE:64bodyforwards;120seconds;frozenoperator.
"""pred_a coefficient formula<=1e-10 and sparse/dense execution<=1e-5.
pred_b selected six-pair own mixed-term margin error<=10% each four groups.
pred_c no sign reversal where native own effect magnitude>=1e-5.
Price64bodyforwards/120seconds; existing frozen sparse packed operator.
Null: regional preservation does not transfer to broader native head ports.
Historical FineWeb prefixes, no fit, no freshOOD or full retained-circuit claim.
"""
import json,os,sys,time,signal
from pathlib import Path
from hashlib import sha256
import torch
import torch.nn.functional as F
import numpy as np
P=Path(__file__).resolve().parents[2]/'polynomial_causal';sys.path.insert(0,str(P))
from head17_output_block_objective_v1 import build
from sparse_interaction_executor_v1 import Executor

@torch.no_grad()
def main():
    files=json.loads((P/'SPARSE_INTERACTION_FINEWEB_V1_BINDING.json').read_text())['files']
    assert all(sha256(Path(k).read_bytes()).hexdigest()==v for k,v in files.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('64bodyforwards;120seconds;frozenoperator');return
    assert not (P/'SPARSE_INTERACTION_FINEWEB_V1_RESULT.json').exists()
    signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
    from fastload import load_model_fast
    model=load_model_fast().cuda().eval();rows=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==64 and all(r['pool']=='fineweb' for r in rows)
    ctx={};states=[]
    def pre17(module,args):ctx['raw']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
    def headinput(module,args):ctx['head']=args[0][:,-1].reshape(-1,9,128)[:,2].double().cpu()
    def afteratt(module,args,output):ctx['z']=(ctx['raw']+output[0])[:,-1].double().cpu()
    handles=[model.transformer.h[17].register_forward_pre_hook(pre17),model.transformer.h[17].attn.c_proj.register_forward_pre_hook(headinput),model.transformer.h[17].attn.register_forward_hook(afteratt)]
    try:
        for row in rows:
            ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
            for block in model.transformer.h:x,v1=block(x,v1,x0)
            states.append((ctx['z'][0],ctx['head'][0],x[0,-1].double().cpu()))
    finally:
        for handle in handles:handle.remove()
    z,a,h=[torch.stack([s[k] for s in states]) for k in range(3)]
    sd=model.state_dict();w=sd['transformer.h.17.attn.c_proj.weight'].double().cpu().reshape(1152,9,128)[:,2]
    extracted=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double();assert torch.equal(w,extracted)
    t,token_ids=build();u=sd['lm_head.weight'][token_ids].double().cpu()
    l,r,d=[sd['transformer.h.17.mlp.'+k+'.weight'].double().cpu() for k in ('Left','Right','Down')]
    v=a@w.T;z0=z-v;exact=torch.einsum('oia,ni,na->no',t,z0,a)
    formula=((z0@l.T)*(v@r.T)+(z0@r.T)*(v@l.T))@(u@d).T
    formula_error=float((formula-exact).norm()/exact.norm())
    program=torch.load(P/'SPARSE_INTERACTION_EXECUTOR_V1_PROGRAM.pt',weights_only=True);executor=Executor(program)
    value=executor(z0.float(),a.float()).double()
    mask=torch.from_numpy(np.unpackbits(program['mask'].numpy(),bitorder='little',count=int(np.prod(program['shape']))).copy()).bool()
    core=torch.zeros(mask.numel(),dtype=torch.float64);core[mask]=program['values'].double()
    fit=torch.einsum('op,pib,ab->oia',program['output'].double(),core.reshape(program['shape']),program['head'].double())
    dense=torch.einsum('oia,ni,na->no',fit,z0,a);replay=float((dense-value).norm()/dense.norm())
    eps=torch.finfo(torch.float32).eps;rhoh=(h.square().mean(-1)+eps).sqrt();denom=(z.square().mean(-1)+eps)*rhoh
    raw=h@u.T/rhoh[:,None];zero=raw-exact/denom[:,None];changed=zero+value/denom[:,None]
    def margins(raw):
        scores=(30*torch.tanh(raw/30)).reshape(64,6,2);return scores[:,:,0]-scores[:,:,1]
    own=margins(raw)-margins(zero);pred=margins(changed)-margins(zero);cells=[]
    for group in range(4):
        ref=own[group*16:(group+1)*16];pp=pred[group*16:(group+1)*16];err=pp-ref
        cells.append(dict(group=group,own_error=float(err.norm()/ref.norm()),reference_norm=float(ref.norm()),max_absolute_error=float(err.abs().max()),material_sign_reversals=int(((pp*ref<0)&(ref.abs()>=1e-5)).sum()),all_sign_reversals=int((pp*ref<0).sum())))
    result={'pred_a':formula_error<=1e-10 and replay<=1e-5,'pred_b':all(c['own_error']<=.1 for c in cells),'pred_c':all(c['material_sign_reversals']==0 for c in cells),
        'cells':cells,'formula_error':formula_error,'executor_replay_error':replay,'raw_mixed_error':float((value-exact).norm()/exact.norm()),'native_own_effects':own.tolist(),'predicted_own_effects':pred.tolist(),'body_forwards':64,'seconds':time.perf_counter()-start,
        'scope':'Historical FineWeb, native full-head17.2 mixed numerator at selected12outputs; frozen sparse model, exact native normalizers/background. Six paired-output probes per prefix, not target-token CE, freshOOD or full retained circuit.'}
    (P/'SPARSE_INTERACTION_FINEWEB_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if not k.endswith('effects')},indent=2))

if __name__=='__main__':main()
