"""Weight-only exact fixed-block selection and node-support accounting.
A replay/energy identity<=1e-10; B best10%error block saves>=10%densebytes;
C some candidate meeting B removes>=10%residual or head reader nodes.
Native/output-head HOSVD frames; outputblocks1/3/6/12, residual/head4/8/16.
No fitting or behavioral claim. Each equal-size block retained by energy order.
"""
import itertools,json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build()
    n=t.numel();total=float(t.square().sum());dense_bytes=4*n
    bases={}
    for axis in (0,2):
        flat=t.movedim(axis,0).reshape(t.shape[axis],-1)
        bases[axis]=torch.linalg.eigh(flat@flat.T)[1].flip(1)
    core=torch.einsum('oa,oid,dc->aic',bases[0],t,bases[2])
    replay=torch.einsum('oa,aic,dc->oid',bases[0],core,bases[2])
    replay_error=float((replay-t).norm()/t.norm());assert replay_error<1e-10
    rows=[];entry=[]
    for name,x,adapters in [('native',t,0),('output_head',core,sum(q.numel() for q in bases.values()))]:
        cumulative=x.flatten().square().sort(descending=True).values.cumsum(0)
        for tol in (.02,.05,.1):
            k=int(torch.searchsorted(cumulative,(1-tol**2)*total))+1
            entry.append(dict(frame=name,tolerance=tol,bytes=4*(k+adapters)+(n+7)//8))
        for bo,bi,bh in itertools.product((1,3,6,12),(4,8,16),(4,8,16)):
            no,ni,nh=12//bo,1152//bi,128//bh
            blocks=x.reshape(no,bo,ni,bi,nh,bh).permute(0,2,4,1,3,5)
            energy=blocks.square().sum((3,4,5));assert abs(float(energy.sum())/total-1)<1e-10
            sorted_energy,order=energy.flatten().sort(descending=True)
            sums=sorted_energy.cumsum(0);blockvolume=bo*bi*bh;blockcount=energy.numel()
            for tol in (.02,.05,.1):
                k=int(torch.searchsorted(sums,(1-tol**2)*total))+1
                mask=torch.zeros(blockcount,dtype=torch.bool);mask[order[:k]]=True;mask=mask.reshape(no,ni,nh)
                active=[int(mask.any(dim=tuple(j for j in range(3) if j!=axis)).sum())*width for axis,width in enumerate((bo,bi,bh))]
                # Rows of a decoded dense tile share its chosen coordinates; no hidden dense adapters omitted.
                cost=4*(k*blockvolume+adapters)+(blockcount+7)//8
                actual=max(0.,1-float(sums[k-1])/total)**.5
                assert actual<=tol+1e-10
                rows.append(dict(frame=name,block_shape=[bo,bi,bh],tolerance=tol,blocks_retained=k,blocks_total=blockcount,
                                 actual_error=actual,bytes=cost,dense_bytes_ratio=cost/dense_bytes,
                                 active_nodes=active,all_nodes=[12,1152,128],
                                 active_input_block_pairs=int(mask.any(dim=0).sum())))
    best=[]
    for tol in (.02,.05,.1):
        chosen=min((r for r in rows if r['tolerance']==tol),key=lambda r:r['bytes'])
        best.append(dict(tolerance=tol,block=chosen,best_entry=min((r for r in entry if r['tolerance']==tol),key=lambda r:r['bytes'])))
    eligible=[r for r in rows if r['tolerance']==.1 and r['dense_bytes_ratio']<=.9]
    result=dict(pred_a=replay_error<=1e-10,pred_b=bool(eligible),
                pred_c=any(r['active_nodes'][1]<=.9*1152 or r['active_nodes'][2]<=.9*128 for r in eligible),
                shape=list(t.shape),token_ids=ids,dense_bytes=dense_bytes,replay_error=replay_error,
                best=best,cells=rows,seconds=time.perf_counter()-start,
                scope='Equal-size blocks in two fixed weight-derived frames. Exact optimal energy support within each tiling, not learned grouping/global graph search. Dense adapters fully charged; bitmap and FP32 values priced. Native-domain fidelity, runtime and wholemodel savings untested.')
    (P/'INTERACTION_BLOCK_SPARSITY_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))


if __name__=='__main__':main()
