"""Conditional regional validation of frozen sparse mixed tensors."""
import json,time
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT
from shared_local_fineweb_v1 import digest
P=Path(__file__).resolve().parent

@torch.no_grad()
def main(zero=False):
    torch.set_num_threads(2);start=time.perf_counter();t,ids=build()
    sd=torch.load(CHECKPOINT,map_location='cpu',mmap=True,weights_only=True)
    u=sd['lm_head.weight'].double()
    l,r,d=[sd['transformer.h.17.mlp.'+k+'.weight'].double() for k in ['Left','Right','Down']]
    w=torch.load(P/'extracted_circuits/three_corner_head17_interaction_v1/program.pt',weights_only=True)['output_matrix'].double()
    cache=P/'COMPOSED_LAST_BLOCK_STATES_V1_ARTIFACT.pt'
    assert digest(cache)==json.loads((P/'COMPOSED_LAST_BLOCK_STATES_V1_RESULT.json').read_text())['artifact_sha']
    data=torch.load(cache,map_location='cpu',weights_only=True)
    z0=data['linear_parts'][:,3].double();z1=data['linear_parts'][:,2].double();v=z1-z0
    a=torch.linalg.lstsq(w,v.T).solution.T;vp=a@w.T
    port_error=float((vp-v).norm()/v.norm())
    mixed=torch.einsum('oia,ni,na->no',t,z0,a)
    direct=((z0@l.T)*(vp@r.T)+(z0@r.T)*(vp@l.T))@(u[ids]@d).T
    formula_error=float((mixed-direct).norm()/direct.norm())
    h=data['states'].double();eps=torch.finfo(torch.float32).eps
    rho=(h.square().mean(-1)+eps).sqrt()
    raw=torch.einsum('nsi,oi->nso',h,u[ids])/rho[:,:,None]
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows'][:24]+json.loads((P/'CROSSFIRST_THREE_GROUP_FRESH_V1_ROWS.json').read_text())['rows']
    lookup={v:k for k,v in enumerate(ids)}
    uk=torch.tensor([lookup[row['uk_id']] for row in rows]);us=torch.tensor([lookup[row['us_id']] for row in rows]);idx=torch.arange(120)
    def margin(x):
        capped=30*torch.tanh(x/30)
        return capped[idx,:,uk]-capped[idx,:,us]
    native=margin(raw);reference=native[:,2]-native[:,0]
    replay=float((native-data['readouts'][:,:,0]).abs().max())
    assert formula_error<=1e-10 and replay<=1e-4 and port_error<=1e-3
    bases=[]
    for axis in range(3):
        f=t.movedim(axis,0).reshape(t.shape[axis],-1)
        bases.append(torch.linalg.eigh(f@f.T)[1])
    results=[];energy=float(t.square().sum());n=t.numel()
    denom=(z1.square().mean(-1)+eps)*rho[:,2]
    for axes in ([[0]] if zero else [[0],[0,2]]):
        core=t.clone();adapters=sum(bases[axis].numel() for axis in axes)
        for axis in axes:core=torch.tensordot(bases[axis].T,core,dims=([1],[axis])).movedim(0,axis)
        values,order=core.flatten().square().sort(descending=True);cum=values.cumsum(0)
        for tol in ([1.0] if zero else [.02,.05,.1]):
            k=0 if zero else int(torch.searchsorted(cum,(1-tol*tol)*energy))+1
            flat=torch.zeros_like(core.flatten());flat[order[:k]]=core.flatten()[order[:k]];fit=flat.reshape_as(core)
            for axis in axes:fit=torch.tensordot(bases[axis],fit,dims=([1],[axis])).movedim(0,axis)
            delta=torch.einsum('oia,ni,na->no',fit-t,z0,a)/denom[:,None]
            changed=raw.clone();changed[:,2]+=delta
            pred=margin(changed)[:,2]-native[:,0];cells=[]
            for group in range(5):
                sl=slice(group*24,(group+1)*24);rr=reference[sl];pp=pred[sl]
                cells.append(dict(group=group,effect_error=float((pp-rr).norm()/rr.norm()),max_absolute_error=float((pp-rr).abs().max()),opposite_signs=int((pp*rr<0).sum()),material_opposite_signs=int(((pp*rr<0)&(rr.abs()>=1e-5)).sum())))
            result=dict(axes=axes,tolerance=tol,coefficient_error=float((fit-t).norm()/t.norm()),retained_entries=k,bytes_ratio=0.0 if zero else (4*adapters+4*k+(n+7)//8)/(4*n),cells=cells,pred_b=all(c['effect_error']<=.05 for c in cells),pred_c=all(c['material_opposite_signs']==0 for c in cells),effects=pred.tolist())
            results.append(result);print(axes,tol,[round(c['effect_error'],5) for c in cells],result['pred_b'],result['pred_c'])
    out=dict(pred_a=True,formula_error=formula_error,port_projection_error=port_error,margin_replay=replay,reference_effects=reference.tolist(),candidates=results,cache_sha256=digest(cache),seconds=time.perf_counter()-start,body_forwards=0,scope='Conditional mixed-numerator replacement only, exact normalization and remaining routes supplied. Historical regional rows; no OOD, state extraction, or control-selectivity claim.')
    name='INTERACTION_SPARSE_REGIONAL_ZERO_V1_RESULT.json' if zero else 'INTERACTION_SPARSE_REGIONAL_V1_RESULT.json'
    with (P/name).open('x') as f:json.dump(out,f,indent=2);f.write('\n')

if __name__=='__main__':main()
