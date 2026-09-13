"""Compact Householder complement coordinates for high-rank interaction cores.
A reflection replay and spectral error agreement<=1e-10.
B each2/5/10%target meets coefficient error.
C strict2%case stores fewer scalars than dense target including adapter.
Weights-only exact input projection; not wholegraph node or speed claim.
"""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);torch.manual_seed(61394);start=time.perf_counter()
    t,ids=build();matrix=t.movedim(1,0).reshape(1152,-1).contiguous()
    eig,basis=torch.linalg.eigh(matrix@matrix.T);total=float(eig.sum())
    rows=[]
    for tol in (.02,.05,.1):
        k=int(torch.searchsorted(eig.clamp_min(0).cumsum(0),tol**2*total,right=True))
        discarded=basis[:,:k].contiguous();packed,tau=torch.geqrf(discarded)
        rotated=torch.ormqr(packed,tau,matrix,left=True,transpose=True)
        core=rotated[k:].contiguous();padded=rotated.clone();padded[:k]=0
        replay=torch.ormqr(packed,tau,padded,left=True,transpose=False)
        expected=matrix-discarded@(discarded.T@matrix)
        identity_error=float((replay-expected).norm()/matrix.norm())
        error=float((replay-matrix).norm()/matrix.norm())
        spectral=(float(eig[:k].sum())/total)**.5
        assert identity_error<=1e-10 and abs(error-spectral)<=1e-10 and error<=tol+1e-10
        # Rebuild only required reflector tails + tau; upper triangle and diagonal of QR are not stored.
        restored=torch.zeros_like(packed)
        for j in range(k):restored[j+1:,j]=packed[j+1:,j]
        restored.diagonal().fill_(1)
        restored_result=torch.ormqr(restored,tau,padded,left=True,transpose=False)
        packed_error=float((restored_result-replay).norm()/matrix.norm());assert packed_error<1e-10
        # Test actual bilinear readout, without materializing a full orthogonal matrix.
        x=torch.randn(1152,7,dtype=t.dtype);h=torch.randn(7,128,dtype=t.dtype)
        coordinates=torch.ormqr(restored,tau,x,left=True,transpose=True)[k:]
        predicted=torch.einsum('ib,ioa,ba->bo',coordinates,core.reshape(1152-k,12,128),h)
        reference=torch.einsum('ib,ioa,ba->bo',x,replay.reshape(1152,12,128),h)
        execution_error=float((predicted-reference).norm()/reference.norm());assert execution_error<1e-10
        adapter_scalars=sum(1152-j for j in range(k))
        rows.append(dict(tolerance=tol,discarded=k,retained=1152-k,actual_error=error,
                         projection_replay_error=identity_error,packed_adapter_error=packed_error,execution_error=execution_error,
                         adapter_scalars=adapter_scalars,core_scalars=core.numel(),total_scalars=adapter_scalars+core.numel(),
                         dense_tensor_scalars=t.numel(),storage_saving=1-(adapter_scalars+core.numel())/t.numel(),
                         native_input_coordinates_still_read=1152,reflector_operations=k))
    result=dict(pred_a=True,pred_b=all(r['actual_error']<=r['tolerance']+1e-10 for r in rows),
                pred_c=rows[0]['storage_saving']>0,rows=rows,token_ids=ids,seconds=time.perf_counter()-start,
                scope='Compact representation of exact optimal input-unfolding projection. Reflector tails and tau counted; implicit leading ones and known shape cost no floating parameters. Dense reduced core retained. Native inputs all read, reflectors are extra operations. No native behavioral/speed/wholemodel adoption.')
    (P/'INTERACTION_COMPLEMENT_ADAPTER_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
