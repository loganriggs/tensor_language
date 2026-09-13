"""Nonorthogonal sparse-core metric discriminator; all metric identities<=1e-10.
Pivot-coordinate chart for an exact projected interaction.
A coefficient/input-executor replay<=1e-10; B FP32readout error<=1e-5;
C packedchart cheaper than reflection representation at2/5/10%error.
PivotedQR chooses coordinate chart, not rank or new fitted subspace.
"""
import json,time
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(61411);start=time.perf_counter();t,ids=build()
    m=t.movedim(1,0).reshape(1152,-1).contiguous();eig,q=torch.linalg.eigh(m@m.T);total=float(m.square().sum())
    previous=json.loads((P/'INTERACTION_COMPLEMENT_ADAPTER_V1_RESULT.json').read_text())['rows'];rows=[]
    for old in previous:
        k=old['discarded'];n=q[:,:k]
        _,_,piv=scipy.linalg.qr(n.T.numpy(),pivoting=True,mode='economic')
        a=torch.as_tensor(np.sort(piv[:k]).copy(),dtype=torch.long)
        mask=torch.zeros(1152,dtype=torch.bool);mask[a]=True;b=(~mask).nonzero().flatten()
        # N_B = C N_A, avoiding an explicit inverse.
        c=torch.linalg.solve(n[a].T,n[b].T).T
        projected=m-n@(n.T@m);core=projected[b].contiguous()
        replay=torch.empty_like(m);replay[b]=core;replay[a]=-c.T@core
        error=float((replay-projected).norm()/m.norm());assert error<1e-10
        x=torch.randn(17,1152,dtype=t.dtype);h=torch.randn(17,128,dtype=t.dtype)
        def execute(x,h,c,core):
            readers=x[:,b]-x[:,a]@c.T
            return torch.einsum('bi,ioa,ba->bo',readers,core.reshape(1152-k,12,128),h)
        expected=torch.einsum('bi,ioa,ba->bo',x,projected.reshape(1152,12,128),h)
        double=execute(x,h,c,core);single=execute(x.float(),h.float(),c.float(),core.float()).double()
        readout_error=float((double-expected).norm()/expected.norm())
        fp32_error=float((single-double).norm()/double.norm())
        assert readout_error<1e-10 and fp32_error<1e-5
        metric_checks=[]
        energies,order=core.flatten().square().sort(descending=True);core_energy=float(core.square().sum())
        for core_tol in (.01,.05,.1):
            keep=int(torch.searchsorted(energies.cumsum(0),(1-core_tol**2)*core_energy))+1
            sparse=torch.zeros_like(core).flatten();sparse[order[:keep]]=core.flatten()[order[:keep]];sparse=sparse.reshape_as(core)
            residual=core-sparse
            naive=float(residual.square().sum());weighted=naive+float((c.T@residual).square().sum())
            approx=torch.empty_like(m);approx[b]=sparse;approx[a]=-c.T@sparse
            lifted=float((approx-projected).square().sum())
            metric_identity_error=abs(lifted-weighted)/max(lifted,1e-30)
            assert metric_identity_error<1e-10
            full_error=float((approx-m).norm()/m.norm())
            partition=(float(eig[:k].sum())/total+weighted/total)**.5
            assert abs(full_error-partition)<1e-10
            metric_checks.append(dict(core_tolerance=core_tol,retained_entries=keep,
                                      naive_relative_to_target=(naive/total)**.5,
                                      lifted_relative_to_target=(weighted/total)**.5,
                                      amplification=(weighted/naive)**.5,
                                      actual_total_error=full_error,metric_identity_error=metric_identity_error))
        cost=4*(c.numel()+core.numel())+144
        rows.append(dict(metric_checks=metric_checks,tolerance=old['tolerance'],discarded=k,retained=1152-k,
                         coefficient_projection_error=float((replay-m).norm()/m.norm()),
                         chart_replay_error=error,readout_error=readout_error,fp32_readout_error=fp32_error,
                         pivot_matrix_condition=float(torch.linalg.cond(n[a])),
                         correction_matrix_spectral_norm=float(torch.linalg.matrix_norm(c,ord=2)),
                         adapter_scalars=c.numel(),core_scalars=core.numel(),mask_bytes=144,
                         total_bytes=cost,dense_storage_saving=1-cost/(4*t.numel()),
                         reflection_total_bytes=4*old['total_scalars'],
                         native_input_coordinates_read=1152))
    result=dict(pred_a=True,pred_b=True,pred_c=all(r['total_bytes']<r['reflection_total_bytes'] for r in rows),
                rows=rows,seconds=time.perf_counter()-start,
                scope='Exact same optimal input projection, alternative coordinate chart. Dense correction/core plus144bytepivotmask, implicitidentity. Arbitrary-inputFP64/FP32readout controls; nativebehavior/runtime untested. Reader metric is nonorthogonal; future sparsecore errors require I+C C.T weighting. All native input coordinates remain read.')
    (P/'INTERACTION_COORDINATE_METRIC_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
