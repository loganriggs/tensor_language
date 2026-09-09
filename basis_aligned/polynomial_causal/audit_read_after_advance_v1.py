"""Opened exact graph-advance/one-hop-reader reuse test; no model or fitting."""
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent;ROWS=BASE/'HOP_QUERY_FACTOR_TRANSFER_V1_ROWS.pt';OUT=BASE/'READ_AFTER_ADVANCE_AUDIT_V1.json'


def main():
    torch.set_num_threads(2);assert not OUT.exists();data=torch.load(ROWS,map_location='cpu',weights_only=True)
    results={};records=[];identity=0.;duplicate=0.
    for pop,block in data.items():
        lookup={}
        for i,m in enumerate(block['metadata']):
            key=(m['world'],m['query'],m['recipient_hop']);value=block['binding_reads']['0'][i]
            if key in lookup:duplicate=max(duplicate,float((value-lookup[key]).abs().max()))
            else:lookup[key]=value
        groups={h:{'target':[],'prediction':[]} for h in (2,3)}
        for world in range(8):
            tok=block['native_tokens'][world*96];f=dict(zip(tok[:48:2].tolist(),tok[1:48:2].tolist()))
            for query in range(24):
                for hop in (1,2,3):
                    address=query
                    for _ in range(hop-1):address=f[address]
                    prediction=lookup[(world,address,1)];target=lookup[(world,query,hop)]
                    if hop==1:identity=max(identity,float((prediction-target).abs().max()));continue
                    prediction=prediction-prediction.mean();target=target-target.mean()
                    relative=float((prediction-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)
                    groups[hop]['target'].append(target);groups[hop]['prediction'].append(prediction)
                    records.append({'population':pop,'world':world,'query':query,'hop':hop,'advanced_query':address,'relative_rms':relative,'passed':relative<=.01})
        results[pop]={}
        for hop,g in groups.items():
            target=torch.stack(g['target']);prediction=torch.stack(g['prediction'])
            relative=float((prediction-target).square().mean().sqrt())/max(float(target.square().mean().sqrt()),1e-6)
            results[pop][str(hop)]={'n':len(target),'relative_rms':relative,'passed':relative<=.01}
    result={'scope':'opened fixed graph-advance/onehop binding-reader equivalence; no fullmodel rewrite or newheldout claim',
        'mechanical_passed':max(identity,duplicate)<=1e-9,'identity_max_abs':identity,'duplicate_native_read_max_abs':duplicate,
        'candidate_passed':all(g['passed'] for p in results.values() for g in p.values()),'populations':results,'cases':records,
        'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'independent_native_constants_retained':387968,'native_coefficients_removed':0}
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
