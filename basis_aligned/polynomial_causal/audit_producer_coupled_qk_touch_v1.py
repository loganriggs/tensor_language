"""Frozen producer-aligned features: measure their joint routing interaction share."""
import hashlib,json,time
from pathlib import Path
import torch
from folded_normalized_router_v1 import rotary
from joint_qk_position_space_v1 import ports
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')


def frames(receipt):
    cache=receipt['cache'];path=Path(cache['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cache['sha256']
    return torch.load(path,weights_only=True,map_location='cpu')['frames']


def main():
    start=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    old=json.loads((P/'POSITION_SHARED_QK_SOURCE_V1_RESULT.json').read_text())
    new=json.loads((P/'MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT.json').read_text())
    routing,producer=frames(old),frames(new)
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    q1,k1,q2,k2=[sd[f'transformer.h.17.attn.{key}.weight'].double().reshape(9,128,1152) for key in ['c_q','c_k','c_q2','c_k2']]
    positions=[0,1,31,63,127,255,383,510]
    rotations=rotary(511,128).T@torch.stack([rotary(s,128) for s in positions])
    rows=[];errors=[]
    for h in range(9):
        base=(q1[h]@q1[h].T,q2[h]@q2[h].T,q1[h]@q2[h].T)
        grams=tuple(rotations.transpose(-1,-2)@g@rotations for g in base)
        before=ports(grams,k1[h],k2[h],routing[h]);after=ports(grams,k1[h],k2[h],producer[h])
        reference=torch.tensor([old['heads'][h]['all_position_touch'][s] for s in positions])
        errors.append(float((reference-before['touch']).abs().max()))
        row=dict(head=h,old_mean_touch=float(before['touch'].mean()),new_mean_touch=float(after['touch'].mean()),
            new_over_old=float(after['touch'].mean()/before['touch'].mean()),
            new_inside=float(after['inside'].mean()),new_mixed=float(after['mixed'].mean()),
            old_function_overlap=new['heads'][h]['frozen_qk_mean'],new_function_overlap=new['heads'][h]['optimal_rank17_mean'],
            positions=positions,new_position_touch=after['touch'].tolist())
        rows.append(row)
    result=dict(instrument_passed=max(errors)<1e-10,maximum_prior_replay_error=max(errors),heads=rows,
        summary=dict(old_mean_touch=sum(row['old_mean_touch'] for row in rows)/9,
            new_mean_touch=sum(row['new_mean_touch'] for row in rows)/9,
            mean_new_over_old=sum(row['new_over_old'] for row in rows)/9,
            producer_alignment_improvement=new['mean_optimal_overlap']/new['mean_frozen_overlap']),
        body_forwards=0,corpus_access=False,wall_seconds=time.perf_counter()-start,
        scope='Post-selection weight diagnostic on8fixed source positions atquery511. Candidate selected for producer alignment only; jointQK1xQK2 numerator touch is separate. No exact-normalized effect, task relevance or full-position guarantee.')
    with (P/'PRODUCER_COUPLED_QK_TOUCH_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':main()
