"""Exact symmetric two-factor partition of cross-context gradient changes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from align_readout_fields import align
from readout_field_differential import contract
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/readout_field_transfer_v1_result.json'
a=json.loads(source.read_text());rows=[];closure=0.
fields={(g['panel'],g['template'],g['role']):g for g in a['fields']}
for (panel,template,role),g in fields.items():
    if panel!='congruent':continue
    donor=fields['opposite',template,role]
    f=torch.tensor(g['fields'],dtype=torch.float64);j=torch.tensor(g['derivatives'],dtype=torch.float64);tokens=torch.tensor(g['tokens']);indices=torch.arange(len(f))
    if role=='attractor':indices=indices^1
    df,dj=align(torch.tensor(donor['fields'],dtype=torch.float64)[indices],torch.tensor(donor['derivatives'],dtype=torch.float64)[indices],torch.tensor(donor['tokens'])[indices],tokens)
    cc,dd,cd,dc=contract(f,j),contract(df,dj),contract(f,dj),contract(df,j)
    change=cc-dd;root=.5*((cc-dc)+(cd-dd));field=.5*((cc-cd)+(dc-dd))
    delta_j=j-dj;num_j=delta_j.clone();num_j[:,-1]=0;rad_j=delta_j-num_j
    numerator=.5*(contract(f,num_j)+contract(df,num_j));radial=.5*(contract(f,rad_j)+contract(df,rad_j))
    closure=max(closure,float((root+field-change).abs().max()),float((root+numerator+radial-change).abs().max()))
    source_rows=json.loads((P/'SOURCE_OOD_V2_CONGRUENT_ROWS.json').read_text());families=[r['family'] for r in source_rows if r['template']==template]
    for family in dict.fromkeys(families):
        ids=[i for i,v in enumerate(families) if v==family]
        for output in range(9):
            delta=change[ids,output];r=root[ids,output];h=field[ids,output];den=float(delta.square().sum())
            rows.append(dict(role=role,family=family,output=output,change_norm=float(delta.norm()),root_norm_fraction=float(r.norm())/max(float(delta.norm()),1e-30),field_norm_fraction=float(h.norm())/max(float(delta.norm()),1e-30),root_projected_share=float((r*delta).sum())/max(den,1e-30),field_projected_share=float((h*delta).sum())/max(den,1e-30),numerator_projected_share=float((numerator[ids,output]*delta).sum())/max(den,1e-30),radial_projected_share=float((radial[ids,output]*delta).sum())/max(den,1e-30)))
number=[r for r in rows if r['output']==0]
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),closure=closure,number_summary=dict(median_root_projected_share=float(np.median([r['root_projected_share'] for r in number])),median_field_projected_share=float(np.median([r['field_projected_share'] for r in number])),median_numerator_projected_share=float(np.median([r['numerator_projected_share'] for r in number])),median_radial_projected_share=float(np.median([r['radial_projected_share'] for r in number])),range_root=[min(r['root_projected_share'] for r in number),max(r['root_projected_share'] for r in number)]),records=rows,scope='Exact algebraic symmetric partition of gradient differences, not causal attribution. Field derivatives include the radial q derivative, so field variation is not synonymous with upstream-only state transport.')
(P/'READOUT_FIELD_CONTEXT_PARTITION_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
