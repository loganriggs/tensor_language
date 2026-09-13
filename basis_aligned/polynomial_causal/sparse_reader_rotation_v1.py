"""Alternating exact hard threshold / orthogonal Procrustes, weights only.
Pred_a monotonic loss and orthogonality <=1e-10.
Pred_b >=10% best loss gain over original gauge at fixed75% support budget.
Pred_c every start tangent-gradient RMS <=1e-7; budget stops are separate.
Ten starts,2000cycles/start,240seconds total; no native behavioral validation.
"""
import json,time
from pathlib import Path
import torch
import numpy as np


def sparse(x, keep):
    mask = torch.zeros(x.numel(), dtype=torch.bool)
    mask[x.abs().flatten().topk(keep, sorted=False).indices] = True
    mask = mask.reshape(x.shape)
    return x*mask,mask


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    start=time.perf_counter();p=Path(__file__).resolve().parent
    n=torch.load(p/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
    b=n['key_basis'][1].double();keep=55296
    s,_=sparse(b,keep);baseline=float((b-s).square().sum());rows=[];best=None
    for arm in range(10):
        gen=torch.Generator().manual_seed(7131340+arm)
        r=torch.eye(64) if arm==0 else torch.linalg.qr(torch.randn(64,64,generator=gen))[0]
        initial=None;previous=float('inf');maxincrease=0.;trace=[]
        for step in range(2000):
            x=b@r;s,mask=sparse(x,keep);loss=float((x-s).square().sum())
            if initial is None:initial=loss
            maxincrease=max(maxincrease,loss-previous);previous=loss
            grad=2*b.T@(x-s);rtg=r.T@grad
            tangent=grad-r@((rtg+rtg.T)/2);stationarity=float(tangent.square().mean().sqrt())
            if step%100==0:trace.append([step,loss,stationarity])
            if stationarity<=1e-7 or time.perf_counter()-start>240:break
            if step==1999:break
            u,_,vh=torch.linalg.svd(b.T@s);r=u@vh
        ortho=float((r.T@r-torch.eye(64)).norm()/8)
        # Round stored sparse values before evaluating the resulting subspace.
        packed=dict(shape=list(s.shape),mask=torch.from_numpy(np.packbits(mask.numpy().flatten(),bitorder='little')),values=s[mask].float())
        from sparse_parent_reader_v1 import unpack
        _,_,q=unpack(packed)
        projector=float((q@q.T-b@b.T).norm()/(b@b.T).norm())
        rows.append(dict(arm=arm,steps=step+1,initial_loss=initial,final_loss=loss,
            stationarity_rms=stationarity,converged=stationarity<=1e-7,
            max_loss_increase=maxincrease,orthogonality_error=ortho,
            projector_relative_error=projector,trace=trace))
        if best is None or loss<best[0]:best=(loss,packed,arm)
        print(json.dumps({k:v for k,v in rows[-1].items() if k!='trace'}),flush=True)
        if time.perf_counter()-start>240:break
    torch.save({'0.25':best[1]},p/'SPARSE_READER_ROTATION_V1_PROGRAM.pt')
    result=dict(pred_a=all(max(x['max_loss_increase'],x['orthogonality_error'])<=1e-10 for x in rows),
        pred_b=best[0]<=.9*baseline,pred_c=len(rows)==10 and all(x['converged'] for x in rows),
        baseline_loss=baseline,best_arm=best[2],relative_loss_gain=1-best[0]/baseline,
        rows=rows,seconds=time.perf_counter()-start,scope=__doc__)
    (p/'SPARSE_READER_ROTATION_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
