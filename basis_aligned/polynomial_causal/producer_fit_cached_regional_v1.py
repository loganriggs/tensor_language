"""Frozen sparse producer-fit cached-native validation, registered10:18UTC.
A FP32 executor<=1e-6; nativeport/replaychecks inherited.
B modeledmixedterm own-effecterror<=10%eachgroup for eachcandidate.
C wholecompacteffecterror<=5%andnomaterialsignreversal eachgroup.
120 historical prefixes, zero-body-forward cache, frozen K32/r8 program.
"""
import json,time
import numpy as np
import scipy.linalg
from pathlib import Path
import torch
from retained_objective_context_v1 import Contexts
from normalized_pair_ht_control_20260914_0256 import CHECKPOINT
from sparse_interaction_executor_v1 import Executor
from shared_local_fineweb_v1 import digest
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent

@torch.no_grad()
def main(program_prefix='INTERACTION_SHARED_WRITE_POLISH_V1', result_prefix='INTERACTION_SHARED_WRITE_REGIONAL_V1'):
    torch.set_num_threads(2);start=time.perf_counter();model=Contexts();t,ids=model.tensor,model.ids
    target=P/(result_prefix+'_RESULT.json');assert not target.exists()
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
    path=P/(program_prefix+'_PROGRAM.pt')
    expected={'SPARSE_INTERACTION_EXECUTOR_V1':'5aefc860fe15be69fb854a4fad3f7c3301f0de6b01c07881f828f027727c730b',
              'UNIFORM_PRODUCER_GRADIENT_V1':'bf5d29ca117cd126ab7c61910b2b1bac74d07874e1e94c36029a9b7ebbad58ed',
              'MINIMAX_ROW_SUPPORT_V1':'538867f86b59d65dedb59b1ff438d5e8a8533d3dc7dd24947f499a532631b432'}
    assert digest(path)==expected[program_prefix]
    program=torch.load(path,map_location='cpu',weights_only=True)
    reconstructed=model.decode(program_prefix)
    actual_coefficient_error=float((reconstructed-t).norm()/t.norm())
    direct_value=torch.einsum('oia,ni,na->no',reconstructed,z0,a)
    value=Executor(program)(z0.float(),a.float()).double()
    execution_error=float((value-direct_value).norm()/direct_value.norm());assert execution_error<=1e-6
    denom=(z1.square().mean(-1)+eps)*rho[:,2]
    zero=raw.clone();zero[:,2]-=mixed/denom[:,None];zero_margin=margin(zero)[:,2]
    own_reference=native[:,2]-zero_margin
    changed=raw.clone();changed[:,2]+=(value-mixed)/denom[:,None]
    changed_margin=margin(changed)[:,2];pred=changed_margin-native[:,0];own_pred=changed_margin-zero_margin
    cells=[]
    for group in range(5):
        sl=slice(group*24,(group+1)*24);rr=reference[sl];pp=pred[sl];own=own_reference[sl];op=own_pred[sl]
        cells.append(dict(group=group,compact_effect_error=float((pp-rr).norm()/rr.norm()),
            own_effect_error=float((op-own).norm()/own.norm()),max_absolute_error=float((pp-rr).abs().max()),
            compact_sign_reversals=int((pp*rr<0).sum()),own_sign_reversals=int((op*own<0).sum()),
            material_compact_sign_reversals=int(((pp*rr<0)&(rr.abs()>=1e-5)).sum()),
            material_own_sign_reversals=int(((op*own<0)&(own.abs()>=1e-5)).sum())))
    result=dict(pred_a=True,pred_b=all(c['own_effect_error']<=.1 for c in cells),
        pred_c=all(c['compact_effect_error']<=.05 and c['material_compact_sign_reversals']==0 and c['material_own_sign_reversals']==0 for c in cells),
        cells=cells,reference_compact_effects=reference.tolist(),reference_own_effects=own_reference.tolist(),
        predicted_compact_effects=pred.tolist(),predicted_own_effects=own_pred.tolist(),
        coefficient_error=actual_coefficient_error,execution_error=execution_error,row_check=row_check,
        port_projection_error=port_error,formula_error=formula_error,margin_replay=replay,
        artifact_sha256=digest(path),cache_sha256=digest(cache),seconds=time.perf_counter()-start,
        scope='Frozen sparse CSR executor,120cachedregionalports, conditionalmixednumerator replacement. Own modeledtermzero reference, native normalizers/background retained; no fitting,freshOOD,fullcircuitextraction orselectivity claim.')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('effects')},indent=2))

if __name__=='__main__':
    import signal
    signal.alarm(120)
    main('SPARSE_INTERACTION_EXECUTOR_V1','PRODUCER_BASELINE_CACHED_REGIONAL_V1')
    main('UNIFORM_PRODUCER_GRADIENT_V1','PRODUCER_FIT_CACHED_REGIONAL_V1')
    baseline=json.loads((P/'PRODUCER_BASELINE_CACHED_REGIONAL_V1_RESULT.json').read_text())
    fit=json.loads((P/'PRODUCER_FIT_CACHED_REGIONAL_V1_RESULT.json').read_text())
    reference=np.array(fit['reference_own_effects'])
    be=np.array(baseline['predicted_own_effects'])-reference
    fe=np.array(fit['predicted_own_effects'])-reference
    improvement=float(1-(fe@fe)/(be@be))
    ratios=[f['own_effect_error']/b['own_effect_error'] for b,f in zip(baseline['cells'],fit['cells'])]
    comparison=dict(relative_squared_error_improvement=improvement,group_error_ratios=ratios,
                    pred_d=improvement>=.05 and max(ratios)<=1.01,
                    scope='Historical native cache; not fresh/OOD. Frozen fit vs original; original synthetic failures remain.')
    path=P/'PRODUCER_FIT_CACHED_COMPARISON_V1_RESULT.json';assert not path.exists()
    path.write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison))
    signal.alarm(0)
