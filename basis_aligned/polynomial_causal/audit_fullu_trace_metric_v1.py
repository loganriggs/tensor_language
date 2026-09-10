"""Red-team output-spectrum negatives using the already-known RMS trace split.

Extends calibration_context_math_audit_v1's scalar trace identity to all outputs.
Separates isotropic input quadratic energy, then recomputes exact output spectra.
No new fitted rank-4 surrogate, no silent replacement of earlier metric/verdict.
"""
import json,time
from pathlib import Path
import torch
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.manual_seed(9116701)
    # A high-rank quadratic can be a constant on the normalized input domain.
    x=torch.randn(64,10,dtype=torch.float64);x=x/x.norm(dim=1,keepdim=True)*10**.5
    toy_error=float((x.square().sum(1)/10-1).abs().max());assert toy_error<=1e-12
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();root=torch.linalg.cholesky(u.T@u);del u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    gram=product_cross(l,r,l,r);trace=(l*r).sum(1);trace_write=d@trace
    raw=d@gram@d.T;weighted=root.T@raw@root
    v=root.T@trace_write;isotropic=v[:,None]*v[None,:]/1152
    traceless=weighted-isotropic
    def summary(matrix):
        ev=torch.linalg.eigvalsh((matrix+matrix.T)/2).flip(0)
        return dict(total_energy=float(ev.sum()),minimum_eigenvalue=float(ev[-1]),
            captured={str(k):float(ev[:k].sum()/ev.sum()) for k in [1,4,32,128,512]})
    full,zero,raw_summary=summary(weighted),summary(traceless),summary(raw)
    fraction=float(isotropic.trace()/weighted.trace())
    # Vector quadratic split on random inputs: B = trace_write*||x||^2/d + B0.
    x=torch.randn(32,1152,dtype=torch.float64)
    native=((x@l.T)*(x@r.T))@d.T
    centered_features=(x@l.T)*(x@r.T)-x.square().sum(1,keepdim=True)*trace[None,:]/1152
    recovered=centered_features@d.T+x.square().sum(1,keepdim=True)*trace_write[None,:]/1152
    replay=float((native-recovered).norm()/native.norm())
    assert replay<=1e-10
    result=dict(schema='fullu.trace.metric.v1',full_coefficient_metric=full,traceless_coefficient_metric=zero,
        residual_output_coefficient_metric=raw_summary,isotropic_coefficient_energy_fraction=fraction,
        uniform_sphere_constant_energy=float(v.square().sum()),
        uniform_sphere_traceless_energy=2*1152/(1152+2)*zero['total_energy'],
        normalized_identity_control_max_error=toy_error,trace_split_replay_error=replay,
        wall_seconds=time.perf_counter()-start,body_forwards=0,
        scope='Exact known trace/radial split; isotropic term retains ||x||^2/d rather than assuming exact unit RMS. Natural states need not be uniform on the sphere. Output spectra are descriptive mathematical bounds in their stated metrics, not behavior or structural impossibility.')
    with (P/'FULLU_TRACE_METRIC_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
