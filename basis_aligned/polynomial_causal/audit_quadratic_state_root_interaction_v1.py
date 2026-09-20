"""Readout generation versus transported state interaction, on the saved surrogate."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from two_axis_state_readout import basis
from final_readout_field_program import evaluate_fields
P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/two_site_composition_v1r1_result.json'
a=json.loads(source.read_text());records=[];closure=0.
for g in a['groups']:
    c={k:torch.tensor(v,dtype=torch.float64) for k,v in g['compiled_core'].items()};b=len(c['numerator'])
    r=torch.zeros(b,6,6,dtype=torch.float64);idx=torch.triu_indices(6,6);r[:,idx[0],idx[1]]=c['norm_triangular']
    def read(phi):
        phi=phi.expand(b,6)
        numerator=torch.einsum('bk,bkv->bv',phi,c['numerator'])
        norm=torch.einsum('bij,bj->bi',r,phi).square().sum(-1,keepdim=True)+torch.finfo(torch.float32).eps
        return evaluate_fields(torch.cat([numerator,norm],dim=-1))
    def phi(s,t):return basis(torch.tensor([s,t],dtype=torch.float64))
    base=read(phi(0,0))
    for s,t in g['coordinates']:
        if not s or not t:continue
        fs,ft,fj=[read(phi(u,v)) for u,v in [(s,0),(0,t),(s,t)]]
        additive=read(phi(s,0)+phi(0,t)-phi(0,0))
        generated=-(additive-fs-ft+base);transported=-(fj-additive);total=fs+ft-base-fj
        closure=max(closure,float((generated+transported-total).abs().max()))
        for family in dict.fromkeys(g['families']):
            ids=[i for i,f in enumerate(g['families']) if f==family];norm=float(total[ids,0].norm())
            records.append(dict(panel=g['panel'],family=family,coordinate=[s,t],total_number_norm=norm,generated_norm=float(generated[ids,0].norm()),transported_norm=float(transported[ids,0].norm()),generated_fraction=float(generated[ids,0].norm())/max(norm,1e-30)))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),closure=closure,median_generated_fraction=float(np.median([r['generated_fraction'] for r in records])),max_generated_fraction=max(r['generated_fraction'] for r in records),records=records,scope='Exact readout split of approximate quadratic state only; native final-state readout split remains unmeasured. Norm fractions are not additive causal shares.')
(P/'QUADRATIC_STATE_ROOT_INTERACTION_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'}))
