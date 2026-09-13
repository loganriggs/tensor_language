"""Frozen coordinate-adapter regional validation.
A exactcoreexecution/reconstruction<=1e-10; nativeport/replaychecks inherited.
B modeledmixedterm own-effecterror<=10%eachgroup for eachcandidate.
C wholecompacteffecterror<=5%andnomaterialsignreversal eachgroup.
120historicalprefixes,zero-body-forward cache,2/5/10%weightonlyprojections.
"""
import json,time
import numpy as np
import scipy.linalg
from pathlib import Path
import torch
from head17_output_block_objective_v1 import build
from head17_source_interface_v1 import CHECKPOINT
from shared_local_fineweb_v1 import digest
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
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
    row_check=validate(rows)
    lookup={v:k for k,v in enumerate(ids)}
    uk=torch.tensor([lookup[row['uk_id']] for row in rows]);us=torch.tensor([lookup[row['us_id']] for row in rows]);idx=torch.arange(120)
    def margin(x):
        capped=30*torch.tanh(x/30)
        return capped[idx,:,uk]-capped[idx,:,us]
    native=margin(raw);reference=native[:,2]-native[:,0]
    replay=float((native-data['readouts'][:,:,0]).abs().max())
    assert formula_error<=1e-10 and replay<=1e-4 and port_error<=1e-3
    matrix=t.movedim(1,0).reshape(1152,-1).contiguous();eig,q=torch.linalg.eigh(matrix@matrix.T)
    energy=float(t.square().sum());denom=(z1.square().mean(-1)+eps)*rho[:,2]
    zero=raw.clone();zero[:,2]-=mixed/denom[:,None];zero_margin=margin(zero)[:,2]
    own_reference=native[:,2]-zero_margin
    candidates=[]
    for tol in (.02,.05,.1):
        k=int(torch.searchsorted(eig.clamp_min(0).cumsum(0),tol**2*energy,right=True));subspace=q[:,:k]
        _,_,piv=scipy.linalg.qr(subspace.T.numpy(),pivoting=True,mode='economic')
        aa=torch.as_tensor(np.sort(piv[:k]).copy(),dtype=torch.long);mask=torch.zeros(1152,dtype=torch.bool);mask[aa]=True;bb=(~mask).nonzero().flatten()
        correction=torch.linalg.solve(subspace[aa].T,subspace[bb].T).T
        projected=matrix-subspace@(subspace.T@matrix);core=projected[bb].reshape(1152-k,12,128)
        readers=z0[:,bb]-z0[:,aa]@correction.T
        value=torch.einsum('ni,ioa,na->no',readers,core,a)
        direct_projected=torch.einsum('ni,ioa,na->no',z0,projected.reshape(1152,12,128),a)
        execution_error=float((value-direct_projected).norm()/direct_projected.norm());assert execution_error<1e-10
        changed=raw.clone();changed[:,2]+=(value-mixed)/denom[:,None]
        changed_margin=margin(changed)[:,2];pred=changed_margin-native[:,0];own_pred=changed_margin-zero_margin
        cells=[]
        for group in range(5):
            sl=slice(group*24,(group+1)*24);rr=reference[sl];pp=pred[sl];own=own_reference[sl];op=own_pred[sl]
            cells.append(dict(group=group,compact_effect_error=float((pp-rr).norm()/rr.norm()),
                              own_effect_error=float((op-own).norm()/own.norm()),
                              own_reference_norm=float(own.norm()),max_absolute_error=float((pp-rr).abs().max()),
                              compact_sign_reversals=int((pp*rr<0).sum()),own_sign_reversals=int((op*own<0).sum()),
                              material_compact_sign_reversals=int(((pp*rr<0)&(rr.abs()>=1e-5)).sum()),
                              material_own_sign_reversals=int(((op*own<0)&(own.abs()>=1e-5)).sum())))
        candidates.append(dict(tolerance=tol,discarded=k,execution_error=execution_error,cells=cells,
            pred_b=all(c['own_effect_error']<=.1 for c in cells),
            pred_c=all(c['compact_effect_error']<=.05 and c['material_compact_sign_reversals']==0 and c['material_own_sign_reversals']==0 for c in cells),
            predicted_compact_effects=pred.tolist(),predicted_own_effects=own_pred.tolist()))
    result=dict(pred_a=True,candidates=candidates,reference_compact_effects=reference.tolist(),reference_own_effects=own_reference.tolist(),
                formula_error=formula_error,port_projection_error=port_error,margin_replay=replay,row_check=row_check,
                cache_sha256=digest(cache),seconds=time.perf_counter()-start,body_forwards=0,
                scope='Frozen coordinateadapter executes conditional mixed numerator only; native RMS/background/otherterms supplied. Own-effect reference removes modeledmixedterm from same rawlogits then reappliessoftcap. Historical120regionalports, nofreshOOD/fullcircuitextraction/selectivity claim.')
    (P/'INTERACTION_COORDINATE_REGIONAL_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in candidates:print(c['tolerance'],c['pred_b'],c['pred_c'],[(round(g['own_effect_error'],5),round(g['compact_effect_error'],5)) for g in c['cells']])
    print('seconds',result['seconds'])

if __name__=='__main__':main()
