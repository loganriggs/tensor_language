"""Replay the exploratory fixed-reader penalty path first executed inline.

Original receipt is preserved; replay writes a separate receipt and verifies it.
No reader optimization, data selection, convergence or circuit claim.
"""
import json
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel
from joint_quadratic_fit_v1 import product_cross
P=Path(__file__).resolve().parent
CK=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')

def main():
    torch.set_num_threads(2)
    state=torch.load(P/'STRUCTURED_FIT_V1_weight_product_s0_CHECKPOINT.pt',map_location='cpu',weights_only=False)
    model=QuadraticModel('product',1152);model.load_state_dict(state['model'])
    with torch.no_grad():a,b,_=model.components()
    gram=product_cross(a,b,a,b)
    sd=torch.load(CK,map_location='cpu',weights_only=True,mmap=True)
    u=sd['lm_head.weight'].double();metric=u.T@u
    l,r,d=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ['Left','Right','Down']]
    cross=d@product_cross(l,r,a,b)
    total=json.loads((P/'STRUCTURED_FIT_STALL_V1_AUDIT.json').read_text())['independent_native_total']
    records=[]
    for penalty in [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,.01]:
        w=torch.linalg.solve(gram+penalty*torch.diag(gram.diag()),cross.T).T
        kg=w.T@metric@w
        error=float((total+(kg*gram).sum()-2*((metric@cross)*w).sum())/total)
        energy=float((kg.diag()*gram.diag()).sum()/total)
        records.append(dict(penalty=penalty,squared_relative_error=error,
            component_energy_over_native_total=energy,cancellation_ratio=float((kg.diag()*gram.diag()).sum()/(kg*gram).sum()),
            optimization_loss=error+penalty*energy))
    original=json.loads((P/'STRUCTURED_FIT_FIXED_READER_PENALTY_V1_AUDIT.json').read_text())['records']
    assert len(original)==len(records)
    discrepancy=max(abs(a[k]-b[k])/max(abs(a[k]),1) for a,b in zip(original,records) for k in a)
    assert discrepancy<=1e-9
    with (P/'STRUCTURED_FIT_FIXED_READER_PENALTY_V1_REPLAY.json').open('x') as f:
        json.dump(dict(records=records,max_scaled_discrepancy=discrepancy),f,indent=2);f.write('\n')
    print(json.dumps(dict(max_scaled_discrepancy=discrepancy)))

if __name__=='__main__':main()
