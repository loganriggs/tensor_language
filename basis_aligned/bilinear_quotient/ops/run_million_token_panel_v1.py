#!/usr/bin/env python3
# BQGATE: 512forwards2048seq512tokens; 1048576tokens; capture65536inputs, no fitting.
"""pred_a exact capture counts; pred_b disjoint document splits; pred_c scaled
FP16 input rounding<=5e-4 and bias-free MLP replay<=1e-3. Null: invalid capture.
Price:512nativebodyforwards2048seq;65536x1152FP16states (~151MB), training
second moment (~11MB). No causal claim. All training positions enter moments;
sampled positions support later non-Gaussian empirical quadratic fitting.
"""
import os,sys,json,time,signal,hashlib
from pathlib import Path
RUNNER=Path(__file__).resolve();ROOT=RUNNER.parents[3];P=ROOT/'basis_aligned/polynomial_causal'
sys.path[:0]=[str(ROOT),str(P)]
import torch
import torch.nn.functional as F
from circuit_fast_screen_producer import Bilin18TorchBackend
from circuit_fast_screen_managed_runner import atomic_create_json
from prepare_million_token_panel_v1 import digest
BIND=P/'MILLION_TOKEN_PANEL_V1_BINDING.json';OUT=P/'MILLION_TOKEN_PANEL_V1_RESULT.json'

def main():
    binding=json.loads(BIND.read_text());assert all(digest(p)==h for p,h in binding.items())
    spec=torch.load(P/'MILLION_TOKEN_PANEL_V1_ROWS.pt',weights_only=True,map_location='cpu')
    source=json.loads((P/'MILLION_TOKEN_PANEL_V1_ROWS.json').read_text())
    rows,pos=spec['rows'],spec['positions'];assert rows.shape==(2048,513) and pos.shape==(2048,32)
    assert int(pos.min())>=0 and int(pos.max())<512
    split_ok=all(torch.equal(spec[k],v) for k,v in [('train_rows',torch.arange(1600)),('validation_rows',torch.arange(1600,1824)),('test_rows',torch.arange(1824,2048))])
    split_ok=split_ok and len({d['document_sha256'] for d in source['documents']})==2048 and len({d['prefix_sha256'] for d in source['documents']})==2048
    assert split_ok
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(model_loaded=False,gpu_accessed=False,forwards=512,sequences=2048,processed_tokens=1048576,sampled_inputs=65536,training_moment_tokens=819200,estimated_artifact_bytes=164000000,split_ok=split_ok)));return
    ap=P/'MILLION_TOKEN_PANEL_V1_INPUTS.pt';assert not OUT.exists() and not ap.exists()
    import shutil
    assert shutil.disk_usage(P).free>200000000,'Need 200MB free before capture'
    signal.alarm(900);tic=time.perf_counter();torch.set_num_threads(2);torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False
    backend=Bilin18TorchBackend.load('cuda');m=backend.model;counts=[0,0];cache={};xs=[];scales=[];rounding=0.;replay=0.
    sum_x=torch.zeros(1152,device='cuda',dtype=torch.float64);sum_xx=torch.zeros(1152,1152,device='cuda',dtype=torch.float64);moment_n=0
    def count(_mod,args):counts[0]+=1;counts[1]+=len(args[0]);assert counts[0]<=512
    def capture(_mod,args,out):cache['x']=args[0];cache['y']=out
    h1=m.transformer.h[0].attn.register_forward_pre_hook(count);h2=m.transformer.h[17].mlp.register_forward_hook(capture)
    try:
        for start in range(0,2048,4):
            idx=rows[start:start+4,:-1].cuda();x=F.rms_norm(m.transformer.wte(idx),(1152,));x0=x;v1=None
            for block in m.transformer.h:x,v1=block(x,v1,x0)
            native_x,native_y=cache['x'],cache['y']
            if start<1600:
                xx=native_x.reshape(-1,1152).float();sum_x+=xx.sum(0,dtype=torch.float64);sum_xx+=(xx.T@xx).double();moment_n+=len(xx)
            bi=torch.arange(4,device='cuda')[:,None];positions=pos[start:start+4].cuda()
            value=native_x[bi,positions].float();assert torch.isfinite(value).all()
            scale=value.abs().amax(-1,keepdim=True).clamp_min(1e-30)/1024
            rounded=(value/scale).half();decoded=rounded.float()*scale
            rounding=max(rounding,float((decoded-value).norm()/value.norm()))
            if start==0:
                target=native_y[bi,positions].float();bias=m.transformer.h[17].mlp.Down_bias.float()
                restored=m.transformer.h[17].mlp(decoded).float()
                replay=float((restored-target).norm()/(target-bias).norm().clamp_min(1e-30))
                assert torch.isfinite(restored).all()
            xs.append(rounded.cpu());scales.append(scale.cpu())
            if start%256==0:print(json.dumps(dict(sequences_completed=start+4,rounding=rounding,replay=replay)),flush=True)
    finally:h1.remove();h2.remove()
    data=dict(x=torch.cat(xs),x_scale=torch.cat(scales),training_mean=(sum_x/moment_n).cpu(),training_second_moment=(sum_xx/moment_n).cpu(),training_moment_tokens=moment_n,**{k:spec[k] for k in ['positions','train_rows','validation_rows','test_rows']})
    valid=counts==[512,2048] and data['x'].shape==(2048,32,1152) and moment_n==819200
    fidelity=rounding<=5e-4 and replay<=1e-3 and bool(torch.isfinite(data['x']).all())
    torch.save(data,ap)
    result=dict(schema='million.token.panel.capture.v1',predictions={'pred_a_capture_counts':bool(valid),'pred_b_document_splits':bool(split_ok),'pred_c_storage_and_regeneration':bool(fidelity)},input_rounding_max_batch_relative_l2=rounding,bias_free_output_regeneration_relative_l2=replay,artifact_sha256=digest(ap),binding_sha256=digest(BIND),runner_sha256=digest(RUNNER),price=dict(body_forwards=counts[0],sequences=counts[1],processed_tokens=1048576,sampled_inputs=65536,training_sampled_inputs=51200,training_moment_tokens=moment_n,artifact_bytes=ap.stat().st_size),wall_seconds=time.perf_counter()-tic,scope='Cached Pile-10k distinct document prefixes, length selection and possible near-duplicates remain. Not verified model-training distribution. Input-only scaledFP16 cache; later target regeneration uses rounded inputs and requires this replay bridge. Training moments use all819200 training positions, FP32 matrix products accumulatedFP64; covariance is not a full fourth-moment fit. Validation/test input capture only; no fit or evaluation on them.')
    atomic_create_json(OUT,result);print(json.dumps(result))
    assert valid and fidelity

if __name__=='__main__':main()
