"""Native source-touch bounds and constructive spectra; fixed rank17, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import torch
from audit_joint_qk_value_ports_v1 import CK,digest
from joint_qk_source_influence_v1 import influence
from joint_qk_source_ports_v1 import source_ports
from folded_normalized_router_v1 import rotary

P=Path(__file__).resolve().parent


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    assert json.loads((P/'JOINT_QK_SOURCE_INFLUENCE_V1_CONTROL.json').read_text())['instrument_passed']
    original=json.loads((P/'JOINT_QK_VALUE_STREAM_CLOSURE_V1_AUDIT.json').read_text())
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    rows=[];errors=[];saved={}
    for h in range(9):
        basis=torch.linalg.qr(torch.cat((k1[h],k2[h]),dim=0).T).Q
        for position in [7,0]:
            rotation=rotary(8,128).T@rotary(position,128)
            ka,kb=rotation@k1[h]@basis,rotation@k2[h]@basis
            s=influence(q1[h],q2[h],ka,kb);eig,vectors=torch.linalg.eigh(s)
            subspace=vectors[:,-17:];parts=source_ports(q1[h],q2[h],ka,kb,subspace)
            total=float(parts['total']);touch=float((parts['inside']+parts['mixed'])/parts['total'])
            bound=min(1.,float(2*eig[-17:].sum()/total))
            errors.extend([abs(float(s.trace())/total-1),max(0.,-float(eig[0])/total),max(0.,touch-bound)])
            before=next(x for x in original['rows'] if x['frame']=='learned' and x['head']==h and x['source_position']==position)
            rows.append(dict(head=h,source_position=position,spectral_touch=touch,rank17_upper_bound=bound,
                             value_space_touch=before['touch'],ratio=touch/before['touch'],
                             top17_influence_fraction=float(eig[-17:].sum()/total),
                             inside_fraction=float(parts['inside']/parts['total']),mixed_fraction=float(parts['mixed']/parts['total'])))
            saved[f'{h}:{position}']=basis@subspace
    mean=sum(x['spectral_touch'] for x in rows)/18;baseline=original['learned_mean_touch'];valid=max(errors)<1e-10
    cache=Path('/dev/shm/bilin18_joint_qk_source_bound_v1.pt');assert not cache.exists();torch.save(saved,cache)
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_choice_of_space_matters':valid and mean>=2*baseline,
                            'pred_c_all_small_spaces_limited':valid and all(x['rank17_upper_bound']<=.25 for x in rows)},
                rows=rows,mean_spectral_touch=mean,mean_value_space_touch=baseline,ratio=mean/baseline,
                maximum_instrument_error=max(errors),wall_seconds=time.perf_counter()-start,body_forwards=0,corpus_access=False,
                cache=dict(path=str(cache),sha256=digest(cache),bytes=cache.stat().st_size,ephemeral=True),
                scope='Constructive rank17 numerator source spaces and upper bounds for all rank17 source projectors. Full normalizers retained; spectra do not establish optimal touch, behavioral circuits or simple value writing.')
    with (P/'JOINT_QK_SOURCE_BOUND_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert valid


if __name__=='__main__':main()
