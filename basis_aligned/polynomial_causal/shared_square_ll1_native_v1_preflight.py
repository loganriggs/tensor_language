"""Weight-only native topology proposals for a shared LL1 square bank."""
import hashlib
import json
import time
from pathlib import Path
import torch
from shared_square_ll1_v1 import canonicalize,merge_readers,execute
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    start=time.perf_counter();torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;rows=[];selected={}
    total=99245061353.47293
    for label in ('spectral','native'):
        source=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
        saved=torch.load(source,weights_only=True,map_location='cpu')
        original=tuple(x.double() for x in saved['parts']);a,s,c=canonicalize(*original)
        q,r=torch.linalg.qr(original[0].transpose(1,2),mode='reduced')
        oldcore=(r*original[1][:,None,:])@r.transpose(1,2)
        transition=q.transpose(1,2)@a.transpose(1,2)
        newcore=(transition*s[:,None,:])@transition.transpose(1,2)
        canonical_error=float((oldcore-newcore).norm()/oldcore.norm())
        oldcp=cp(a,s,c);old_floats=sum(v.numel() for v in original)
        candidates=[]
        for threshold in (.995,.999,.9999):
            readers,indices,weights,clusters=merge_readers(a,s,c,threshold)
            newcp=cp(readers[indices],weights,c)
            delta=(torch.cat((newcp[0],oldcp[0])),torch.cat((newcp[1],oldcp[1])),torch.cat((newcp[2],-oldcp[2]),1))
            change=float(inner(delta,delta)/total)
            torch.manual_seed(2302);x=torch.randn(11,a.shape[-1])
            reference=((x@newcp[0].T)*(x@newcp[1].T))@newcp[2].T
            replay=float((execute(readers,indices,weights,c,x)-reference).norm()/reference.norm())
            floats=readers.numel()+weights.numel()+c.numel();savings=old_floats-floats
            row=dict(label=label,threshold=threshold,unique_readers=len(readers),reader_nodes_removed=a.shape[0]*a.shape[1]-len(readers),
                     old_floats=old_floats,new_floats=floats,float_savings=savings,float_savings_fraction=savings/old_floats,
                     graph_indices=indices.numel(),net_bytes_saved_float32_matrices_int64_indices=4*savings-8*indices.numel(),
                     squared_surrogate_change_over_native=change,maximum_cluster_size=max(map(len,clusters)),
                     reused_clusters=sum(len(cl)>1 for cl in clusters),canonicalization_replay=canonical_error,executor_replay=replay,
                     pred_a=max(canonical_error,replay)<1e-9,
                     pred_b=savings/old_floats>=.05 and a.shape[0]*a.shape[1]-len(readers)>=16 and change<=1e-4)
            rows.append(row);print(json.dumps(row),flush=True)
            candidates.append((row,dict(readers=readers,indices=indices,s=weights,c=c,clusters=clusters)))
        eligible=[item for item in candidates if item[0]['pred_a'] and item[0]['pred_b']]
        if eligible:
            row,program=max(eligible,key=lambda item:item[0]['float_savings'])
            selected[label]=dict(**program,threshold=row['threshold'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    cache=Path('/dev/shm/bilin18_shared_square_ll1_v1_preflight.pt')
    torch.save(selected,cache)
    result=dict(pred_a=all(r['pred_a'] for r in rows),pred_b=len(selected)==2,rows=rows,
                selected_thresholds={k:v['threshold'] for k,v in selected.items()},
                cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),seconds=time.perf_counter()-start,
                scope='Canonicalize each existing LL1 quadratic, propose complete-link shared square bank, score exact frozen-surrogate change. No native optimization, corpus access, global graph-recovery or behavioral claim.')
    (root/'SHARED_SQUARE_LL1_NATIVE_V1_PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
