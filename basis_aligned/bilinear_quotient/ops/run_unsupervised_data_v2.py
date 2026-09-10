#!/usr/bin/env python3
# BQGATE: 250forwards1000seq512tokens; capture64000unlabeledstates; no fitting.
"""pred_a instrument/counts; pred_b row-split integrity; pred_c FP16 relative
capture rounding <=5e-4 and finite. Dataset preparation, not a circuit screen.
"""
import os, sys, json, time, signal, hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(POLY)]
import torch
import torch.nn.functional as F
from circuit_fast_screen_producer import Bilin18TorchBackend
from circuit_fast_screen_managed_runner import atomic_create_json
from induction_context_transport_v2 import digest
OUT=POLY/'UNSUPERVISED_DATA_V2_RESULT.json';BIND=POLY/'UNSUPERVISED_DATA_V2_BINDING.json'

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    spec=torch.load(POLY/'UNSUPERVISED_DATA_V1_ROWS.pt',map_location='cpu',weights_only=True)
    rows,pos=spec['rows'],spec['positions'];assert rows.shape==(1000,513) and pos.shape==(1000,64)
    splits=[spec[k] for k in ['train_rows','validation_rows','test_rows']]
    hashes=[hashlib.sha256(r.numpy().tobytes()).hexdigest() for r in rows]
    split_ok=sorted(torch.cat(splits).tolist())==list(range(1000)) and len(set(hashes))==1000
    assert split_ok and [len(x) for x in splits]==[800,100,100] and int(pos.min())>=64 and int(pos.max())<512
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,forwards=250,sequences=1000,processed_tokens=512000,sampled_states=64000,estimated_state_bytes=294912000,split_ok=split_ok)));return
    assert not OUT.exists();signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    backend=Bilin18TorchBackend.load('cuda');m=backend.model;counts=[0,0];cache={};xs=[];ys=[];scales={'x':[],'y':[]};rounding={};maxabs={}
    def count(_mod,args):counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=250
    def capture(_mod,args,out):cache['x']=args[0];cache['y']=out
    h1=m.transformer.h[0].attn.register_forward_pre_hook(count);h2=m.transformer.h[17].mlp.register_forward_hook(capture)
    try:
        for start in range(0,1000,4):
            idx=rows[start:start+4,:-1].cuda();x=F.rms_norm(m.transformer.wte(idx),(1152,));x0=x;v1=None
            for block in m.transformer.h:x,v1=block(x,v1,x0)
            positions=pos[start:start+4].cuda();bi=torch.arange(4,device='cuda')[:,None]
            for key,store in [('x',xs),('y',ys)]:
                value=cache[key][bi,positions].float();assert torch.isfinite(value).all(), 'Native nonfinite '+key
                scale=value.abs().amax(-1,keepdim=True).clamp_min(1e-30)/1024
                rounded=(value/scale).half();assert torch.isfinite(rounded).all()
                scales[key].append(scale.cpu())
                error=float((rounded.float()*scale-value).norm()/value.norm().clamp_min(1e-30))
                rounding[key]=max(rounding.get(key,0.),error);maxabs[key]=max(maxabs.get(key,0.),float(value.abs().max()))
                store.append(rounded.cpu())
            if start%200==0:print(json.dumps(dict(sequences_completed=start+4,rounding=rounding)),flush=True)
    finally:h1.remove();h2.remove()
    data=dict(x=torch.cat(xs),y=torch.cat(ys),x_scale=torch.cat(scales['x']),y_scale=torch.cat(scales['y']),**{k:spec[k] for k in ['positions','train_rows','validation_rows','test_rows']})
    ap=POLY/'UNSUPERVISED_DATA_V2_STATES.pt';assert not ap.exists();torch.save(data,ap)
    valid=counts==[250,1000] and data['x'].shape==data['y'].shape==(1000,64,1152)
    out=dict(schema='unsupervised.data.v2',predictions={'pred_a_capture_instrument':bool(valid),'pred_b_split_integrity':bool(split_ok),'pred_c_storage_fidelity':bool(valid and max(rounding.values())<=5e-4)},
        rounding_max_batch_relative_l2=rounding,native_max_abs=maxabs,shape=list(data['x'].shape),splits=dict(train=51200,validation=6400,test=6400),
        row_sha256=hashes,artifact_sha256=digest(ap),runner_sha256=digest(RUNNER),binding_sha256=digest(BIND),
        price=dict(body_forwards=counts[0],sequences=counts[1],processed_tokens=512000,state_pairs=64000,artifact_bytes=ap.stat().st_size),
        wall_seconds=time.perf_counter()-tic,scope='Unlabeled MLP17 normalized inputs and native outputs from historically opened cached corpus rows. Row-held-out splits, not fresh/OOD or verified document-level separation. Scaled FP16: decode x*x_scale and y*y_scale in FP32. Same relative rounding bar; V1 direct-half overflow preserved.')
    atomic_create_json(OUT,out);print(json.dumps({k:v for k,v in out.items() if k!='row_sha256'}))

if __name__=='__main__':main()
