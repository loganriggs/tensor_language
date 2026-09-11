"""Post-selected opening-parenthesis association, cached states only.

A cache/positions valid; B >=75% eligible rows have lower mean at '(';
C median within-row difference/global native RMS<=-.5. No causal inference.
"""
import hashlib,json
from pathlib import Path
import torch
from tokenizers import Tokenizer


def main():
    p=Path(__file__).resolve().parent;out=p/'SHARED_FUNCTION_PUNCTUATION_V1_AUDIT.json';assert not out.exists()
    parent=json.loads((p/'SHARED_FUNCTION_INTERFACE_V1_AUDIT.json').read_text())
    validation=json.loads((p/'SHARED_FUNCTION_FINEWEB_V1_AUDIT.json').read_text())
    sources=[validation['cache'],validation['data_source'],parent['tokenizer']]
    for s in sources:assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256']
    scalar=torch.load(sources[0]['path'],weights_only=True,map_location='cpu')['native'].reshape(64,128)
    tokens=torch.load(sources[1]['path'],weights_only=True,map_location='cpu')['rows'][:,:128]
    tokenizer=Tokenizer.from_file(sources[2]['path'])
    ids=[int(i) for i in tokens.unique() if tokenizer.decode([int(i)]).strip()=='(']
    mask=torch.zeros_like(tokens,dtype=torch.bool)
    for i in ids:mask|=tokens==i
    rms=scalar.square().mean().sqrt();rows=[]
    for i in range(64):
        if mask[i].any() and (~mask[i]).any():
            a=scalar[i,mask[i]].mean();b=scalar[i,~mask[i]].mean()
            rows.append(dict(row=i,parenthesis_positions=int(mask[i].sum()),parenthesis_mean=float(a),
                other_mean=float(b),difference_in_global_rms=float((a-b)/rms)))
    differences=torch.tensor([r['difference_in_global_rms'] for r in rows]);assert len(rows)>0
    lower=float((differences<0).double().mean());median=float(differences.median())
    result=dict(predictions=dict(pred_a_instrument=tokens.shape==scalar.shape==(64,128) and len(rows)>0,
        pred_b_consistent_association=lower>=.75,pred_c_large_association=median<=-.5),
        eligible_rows=len(rows),parenthesis_positions=int(mask.sum()),fraction_rows_lower=lower,
        median_difference_in_native_rms=median,token_ids=ids,rows=rows,sources=sources,
        scope='Post-selected observational within-row association. No token-edit experiment, matched causal control, clean gender semantics, fresh data or factor fitting.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','sources']},indent=2))


if __name__=='__main__':main()
