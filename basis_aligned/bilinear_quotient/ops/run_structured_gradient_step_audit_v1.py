#!/usr/bin/env python3
"""pred_a gradientreplay<=1e-10; pred_b errorratio>=20; pred_c Richardson<=1e-5.

BQGATE:0forwards0seq. Reuse savednativeinitialization; no optimization.
"""
import hashlib,json,os,sys,time
from pathlib import Path
RUNNER=Path(__file__).resolve();P=RUNNER.parents[3]/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from structured_bilinear_bank_v1 import StructuredBank
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    binding=json.loads((P/'STRUCTURED_GRADIENT_STEP_AUDIT_V1_BINDING.json').read_text())
    assert all(digest(f)==h for f,h in binding.items())
    if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
        print(json.dumps(dict(gpu_accessed=False,body_forwards=0,corpus_access=False,optimization=False)));return
    out=P/'STRUCTURED_GRADIENT_STEP_AUDIT_V1_RESULT.json';assert not out.exists()
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    start=time.perf_counter();prior=json.loads((P/'STRUCTURED_BILINEAR_NATIVE_V1_PREFLIGHT_0.json').read_text())
    saved=torch.load(prior['initial_cache']['path'],map_location='cpu',weights_only=True)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double().cuda();whitener=torch.linalg.cholesky(u.T@u).T;del u
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double().cuda() for key in ('Left','Right','Down')]
    native=(l,r,whitener@d);total=json.loads((P/'FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json').read_text())['native_total']
    model=StructuredBank([2]*7+[3,3]).cuda();model.load_state_dict(saved['model']);model.zero_grad(set_to_none=True)
    loss,_=model.coefficient_loss(native,whitener,total)
    params=list(model.parameters());point=[p.detach().clone() for p in params]
    gn=sum(p.grad.square().sum() for p in params).sqrt()
    pn=sum(p.detach().square().sum() for p in params).sqrt().clamp_min(1)
    direction=[p.grad.detach().clone()/gn*pn for p in params];expected=float(gn*pn)
    rows=[]
    for eps in (1e-4,3e-5,1e-5,5e-6,1e-6):
        values=[]
        for sign in (1,-1):
            with torch.no_grad():
                for p,x,v in zip(params,point,direction):p.copy_(x+sign*eps*v)
            values.append(float(model.coefficient_loss(native,whitener,total,backward=False)[0]))
        finite=(values[0]-values[1])/(2*eps)
        row=dict(eps=eps,derivative=finite,relative_error=abs(finite-expected)/abs(expected))
        rows.append(row);print(json.dumps(row),flush=True)
    with torch.no_grad():
        for p,x in zip(params,point):p.copy_(x)
    rich=(4*rows[3]['derivative']-rows[2]['derivative'])/3
    rich_error=abs(rich-expected)/abs(expected)
    ratio=rows[0]['relative_error']/max(rows[2]['relative_error'],1e-30)
    replay=abs(expected-prior['parameter_fd_expected'])/abs(expected)
    result=dict(predictions={'pred_a_replay':replay<=1e-10,'pred_b_truncation_scaling':ratio>=20,
                            'pred_c_richardson':rich_error<=1e-5},
                expected=expected,replay_error=replay,steps=rows,error_ratio=ratio,
                richardson_derivative=rich,richardson_relative_error=rich_error,
                wall_seconds=time.perf_counter()-start,binding=binding,body_forwards=0,corpus_access=False,
                scope='Savednativepoint derivative instrument audit; no optimization or nativefit verdict.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('binding','steps')},indent=2),flush=True)


if __name__=='__main__':main()
