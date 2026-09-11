"""Common versus centered coefficient capture for the new23x4native fits."""
import hashlib
import json
from pathlib import Path
import torch
from conditional_block_svd_v1 import pack,project_core

torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
p=Path(__file__).resolve().parent
source=json.loads((p/'FULL_QUADRATIC_FRAME_V1_RESULT.json').read_text())
totals=json.loads((p/'SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json').read_text())['metrics']
full=totals['full']['total_energy'];centered=totals['centered']['total_energy'];common=full-centered
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
sd=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
u=sd['lm_head.weight'];vocab=len(u);mean=u.double().mean(0)
l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
native_mean_writer=mean@d
rows=[]
for entry in source['starts']:
    path=Path(entry['cache']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['cache']['sha256']
    saved=torch.load(path,map_location='cpu',weights_only=True)
    bank=saved['bank'];writer=saved['writer'];groups=len(bank)
    coefficient=(mean@writer).reshape(groups,10)
    norm=0.;dot=0.
    for g in range(groups):
        a,b=l@bank[g],r@bank[g]
        native=pack((a[:,:,None]*b[:,None,:]+b[:,:,None]*a[:,None,:])/2)
        dot+=float((native_mean_writer@native)@coefficient[g])
        for h in range(groups):
            transported=project_core(torch.eye(10),bank[h].T@bank[g])
            norm+=float(coefficient[g]@transported@coefficient[h])
    common_error=(common+vocab*(norm-2*dot))/common
    centered_error=(entry['final']['residual']*full-common_error*common)/centered
    orth=float((bank.transpose(-1,-2)@bank-torch.eye(4)).abs().max())
    packed_error=float((saved['packed']-torch.eye(10).repeat(1,groups)).abs().max())
    rows.append(dict(seed=entry['seed'],full_capture=entry['capture'],common_capture=1-common_error,
                     centered_capture=1-centered_error,orthogonality_error=orth,packing_error=packed_error,
                     common_contribution_to_full_capture=vocab*(2*dot-norm)/full))
result=dict(predictions={'pred_a_frame_and_packing':max(max(r['orthogonality_error'],r['packing_error']) for r in rows)<=1e-8,
                        'pred_b_both_common_half':all(r['common_capture']>=.5 for r in rows),
                        'pred_c_both_centered_below_five_percent':all(r['centered_capture']<.05 for r in rows)},
    starts=rows,native_common_fraction=common/full,body_forwards=0,corpus_access=False,gpu_access=False,
    scope='New fit decomposition in coefficient metric. Common output before native normalization/tanh is not a discardable softmax constant. No independent native-total recomputation.')
with (p/'FULL_QUADRATIC_OUTPUT_SPLIT_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2));assert result['predictions']['pred_a_frame_and_packing']
