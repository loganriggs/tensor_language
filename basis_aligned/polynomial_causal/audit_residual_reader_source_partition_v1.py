"""Exact symmetric partition of G=R D across source-number-matched contexts."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent;A=P.parent/'bilinear_quotient/circuits/followups';source=A/'residual_reader_transfer_v1_result.json';a=json.loads(source.read_text());tensors=A/'residual_reader_transfer_v1_tensors.pt';assert hashlib.sha256(tensors.read_bytes()).hexdigest()==a['tensor_sha256']
bundle=torch.load(tensors,map_location='cpu',weights_only=True);groups={(g['panel'],g['template'],g['role']):g for g in bundle['groups']};records=[];closure=0.
contract=lambda r,d:torch.einsum('bod,bkd->bok',r,d)
for (panel,template,role),current in groups.items():
    if panel!='congruent':continue
    donor=groups['opposite',template,role];ids=torch.arange(len(current['reader']))
    if role=='attractor':ids=ids^1
    rc,dc=current['reader'],current['sources'];rd,dd=donor['reader'][ids],donor['sources'][ids]
    difference=contract(rc,dc)-contract(rd,dd)
    reader=.5*(contract(rc-rd,dc)+contract(rc-rd,dd));source_part=.5*(contract(rc+rd,dc-dd))
    closure=max(closure,float((reader+source_part-difference).abs().max()))
    for family in dict.fromkeys(current['families']):
        ids=[i for i,f in enumerate(current['families']) if f==family]
        for output in range(9):
            delta=difference[ids,output];den=float(delta.square().sum());r=reader[ids,output];d=source_part[ids,output]
            records.append(dict(role=role,family=family,output=output,change_norm=float(delta.norm()),reader_projected_share=float((r*delta).sum())/max(den,1e-30),source_projected_share=float((d*delta).sum())/max(den,1e-30),reader_norm_fraction=float(r.norm()/delta.norm().clamp_min(1e-30)),source_norm_fraction=float(d.norm()/delta.norm().clamp_min(1e-30))))
summary=[]
for role in ['subject','attractor']:
    rr=[r for r in records if r['role']==role and r['output']==0]
    summary.append(dict(role=role,median_reader_share=float(np.median([r['reader_projected_share'] for r in rr])),median_source_share=float(np.median([r['source_projected_share'] for r in rr])),reader_share_range=[min(r['reader_projected_share'] for r in rr),max(r['reader_projected_share'] for r in rr)]))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),closure=closure,summary=summary,records=records,scope='Canonical is-minus-are gradient coordinates; exact algebraic partition, not finite causal attribution. Readers include all suffix nonlinear derivatives; source vectors include prefix response variation.')
(P/'RESIDUAL_READER_SOURCE_PARTITION_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
