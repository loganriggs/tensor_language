"""Native weights-only joint QK/value source-port audit; CPU two threads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as functional
from joint_qk_source_ports_v1 import source_ports
from folded_normalized_router_v1 import rotary,EPS

P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    assert json.loads((P/'JOINT_QK_SOURCE_PORTS_V1_CONTROL.json').read_text())['instrument_passed']
    source=json.loads((P/'JOINT_SHARED_READER_RANK16_V1_RESULT.json').read_text());cache=source['cache']
    assert digest(cache['path'])==cache['sha256']
    selected=torch.load(cache['path'],weights_only=True,map_location='cpu')['programs'][source['best']]
    learned=torch.cat((selected['reader'][None,:],selected['partner_readers']),dim=0)
    learned=torch.linalg.qr(learned.T).Q.T
    frames={'learned':learned}
    for seed in [1213,1217,1223]:
        torch.manual_seed(seed);frames[str(seed)]=torch.linalg.qr(torch.randn(1152,17)).Q.T
    sd=torch.load(CK,map_location='cpu',mmap=True,weights_only=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    o=sd['transformer.h.17.attn.c_proj.weight'].double()
    vc=sd['transformer.h.17.attn.c_v.weight'].double();vb=sd['transformer.h.0.attn.c_v.weight'].double()
    mix=float(sd['transformer.h.17.attn.lamb']);values=torch.cat(((1-mix)*vc,mix*vb),dim=1)
    query_rotation=rotary(8,128);rows=[];errors=[]
    torch.manual_seed(1229);query=torch.randn(23,1152);state=torch.randn(23,2304)
    for label,frame in frames.items():
        for h in range(9):
            sl=slice(h*128,(h+1)*128);ports=frame@o[:,sl]@values[sl]
            _,sing,right=torch.linalg.svd(ports,full_matrices=False)
            rank=int((sing>sing[0]*1e-10).sum());assert rank==17
            basis=right[:rank].T;errors.append(float((basis.T@basis-torch.eye(rank)).abs().max()))
            for position in [7,0]:
                rs=rotary(position,128);rotation=query_rotation.T@rs
                ka,kb=[torch.cat((rotation@k[h],torch.zeros(128,1152)),dim=1) for k in [k1,k2]]
                energy=source_ports(q1[h],q2[h],ka,kb,basis)
                fractions={key:float(energy[key]/energy['total']) for key in ['inside','mixed','outside']}
                errors.append(max(0.,-min(fractions.values()),max(fractions.values())-1))
                rows.append(dict(frame=label,head=h,source_position=position,source_rank=rank,
                                 fractions=fractions,touch=fractions['inside']+fractions['mixed']))
                if label=='learned':
                    inside_state=(state@basis)@basis.T;outside_state=state-inside_state
                    qa,qb=query@q1[h].T,query@q2[h].T
                    ia,ib=inside_state@ka.T,inside_state@kb.T
                    oa,ob=outside_state@ka.T,outside_state@kb.T
                    dot=lambda x,y:(x*y).sum(-1)
                    numerator=dot(qa,ia)*dot(qb,ib)+dot(qa,oa)*dot(qb,ob)+dot(qa,ia)*dot(qb,ob)+dot(qa,oa)*dot(qb,ib)
                    keys=state[:,:1152];kha,khb=keys@k1[h].T,keys@k2[h].T
                    denominator=((qa.square().mean(-1)+EPS)*(qb.square().mean(-1)+EPS)*(kha.square().mean(-1)+EPS)*(khb.square().mean(-1)+EPS)).sqrt()
                    folded=numerator/(128**2*denominator)
                    branches=[]
                    for q,k in [(qa,kha),(qb,khb)]:
                        qr=functional.rms_norm(q,(128,),eps=EPS)@query_rotation.T
                        kr=functional.rms_norm(k,(128,),eps=EPS)@rs.T
                        branches.append(dot(qr,kr)/128)
                    direct=branches[0]*branches[1]
                    errors.append(float((folded-direct).norm()/direct.norm()))
    learned_rows=[r for r in rows if r['frame']=='learned'];random_rows=[r for r in rows if r['frame']!='learned']
    lm=sum(r['touch'] for r in learned_rows)/18;rm=sum(r['touch'] for r in random_rows)/54
    valid=max(errors)<1e-10
    result=dict(predictions={'pred_a_instrument':valid,'pred_b_selected_alignment':valid and lm>=2*rm and lm-rm>=.02,
                            'pred_c_focus_pair_substantial_touch':valid and all(r['touch']>=.1 for r in learned_rows if r['head'] in [2,3])},
                learned_mean_touch=lm,random_mean_touch=rm,learned_to_random_ratio=lm/rm,absolute_touch_gain=lm-rm,
                rows=rows,maximum_instrument_error=max(errors),source_cache_sha256=cache['sha256'],
                source_sha256=digest(__file__),preregistration_sha256=digest(P/'JOINT_QK_VALUE_PORTS_V1_PREREGISTRATION.md'),
                helper_sha256=digest(P/'joint_qk_source_ports_v1.py'),
                wall_seconds=time.perf_counter()-start,body_forwards=0,corpus_access=False,
                scope='Exact joint QK numerator inside/mixed/outside source ports, conditional on a frozen compact MLP component. Full normalization retained. Distinct query/source positions only. No behavioral or model-level adoption claim.')
    with (P/'JOINT_QK_VALUE_PORTS_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    assert valid


if __name__=='__main__':main()
