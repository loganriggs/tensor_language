"""Executed red-team: separate current/base value streams before source overlap.

The enlarged space changes the representation. It does not rewrite V1's bars.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
import torch
from audit_joint_qk_value_ports_v1 import CK,digest
from joint_qk_source_ports_v1 import source_ports
from folded_normalized_router_v1 import rotary

P=Path(__file__).resolve().parent


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    original=json.loads((P/'JOINT_QK_VALUE_PORTS_V1_RESULT.json').read_text())
    source=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text());cache=source['cache']
    assert digest(cache['path'])==cache['sha256']
    selected=torch.load(cache['path'],weights_only=True,map_location='cpu')['programs'][source['best']]
    frame=torch.cat((selected['reader'][None,:],selected['partner_readers']),dim=0)
    frames={'learned':torch.linalg.qr(frame.T).Q.T}
    for seed in [1213,1217,1223]:
        torch.manual_seed(seed);frames[str(seed)]=torch.linalg.qr(torch.randn(1152,17)).Q.T
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double()
    vc=sd['transformer.h.17.attn.c_v.weight'].double()
    vb=sd['transformer.h.0.attn.c_v.weight'].double()
    mix=float(sd['transformer.h.17.attn.lamb']);rows=[];errors=[]
    for label,frame in frames.items():
        for h in range(9):
            sl=slice(h*128,(h+1)*128);intermediate=frame@o[:,sl]
            current=(1-mix)*intermediate@vc[sl];base=mix*intermediate@vb[sl]
            _,s,v=torch.linalg.svd(current,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==17
            basis=v[:rank].T
            _,bs,_=torch.linalg.svd(base,full_matrices=False);base_rank=int((bs>bs[0]*1e-10).sum())
            errors.append(float((basis.T@basis-torch.eye(rank)).abs().max()))
            for position in [7,0]:
                rotation=rotary(8,128).T@rotary(position,128)
                energy=source_ports(q1[h],q2[h],rotation@k1[h],rotation@k2[h],basis)
                frac={key:float(energy[key]/energy['total']) for key in ['inside','mixed','outside']}
                before=next(x for x in original['rows'] if x['frame']==label and x['head']==h and x['source_position']==position)
                touch=frac['inside']+frac['mixed']
                errors.append(max(0.,before['touch']-touch))
                rows.append(dict(frame=label,head=h,source_position=position,fractions=frac,touch=touch,
                                 original_touch=before['touch'],closure_touch_gain=touch-before['touch'],
                                 source_space_rank=rank+base_rank,
                                 current_map_energy_fraction=float(current.square().sum()/(current.square().sum()+base.square().sum()))))
    head_rows=[]
    for h in range(9):
        learned=[x for x in rows if x['head']==h and x['frame']=='learned']
        null=[x for x in rows if x['head']==h and x['frame']!='learned']
        lm=sum(x['touch'] for x in learned)/2;rm=sum(x['touch'] for x in null)/6
        head_rows.append(dict(head=h,learned_mean_touch=lm,random_mean_touch=rm,ratio=lm/rm,
                              original_learned_mean_touch=sum(x['original_touch'] for x in learned)/2,
                              learned_current_map_energy_fraction=learned[0]['current_map_energy_fraction']))
    lm=sum(x['learned_mean_touch'] for x in head_rows)/9;rm=sum(x['random_mean_touch'] for x in head_rows)/9
    result=dict(instrument_passed=max(errors)<1e-10,maximum_instrument_error=max(errors),head_summary=head_rows,
                learned_mean_touch=lm,random_mean_touch=rm,ratio=lm/rm,
                enlarged_space_meets_original_alignment_numbers=lm>=2*rm and lm-rm>=.02,
                enlarged_space_focus_pair_meets_original_numbers=all(x['touch']>=.1 for x in rows if x['frame']=='learned' and x['head'] in [2,3]),
                rows=rows,wall_seconds=time.perf_counter()-start,
                original_predictions=original['predictions'],body_forwards=0,corpus_access=False,
                scope='Post-result stream-closure and head-resolved audit. Up to34 source directions instead of17; null frames enlarged identically. Original failures preserved. No broad negative about routing/value or task-specific sharing.')
    with (P/'JOINT_QK_VALUE_STREAM_CLOSURE_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
