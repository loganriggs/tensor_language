"""Prospective support/price audit, using token identities and no model effects."""
import hashlib,json
from pathlib import Path
import torch
import token_local_attention_support as T
P=Path(__file__).resolve().parent
SOURCE=P/'SEMANTIC_SQUARE_ROWS_V1_AUDIT.json'
OUT=P/'TOKEN_LOCAL_ATTENTION_SUPPORT_V1_ROW_AUDIT.json'


def main():
    data=json.loads(SOURCE.read_text());panels={}
    for panel,entry in data['panels'].items():
        records=[]
        for s in entry['squares']:
            n=s['token_length'];cue=torch.zeros(n,dtype=torch.bool);context=cue.clone()
            cue[s['cue_positions']]=True;context[s['context_positions']]=True
            single=T.single_edit_edges(cue);cross=T.mixed_edit_edges(cue,context)
            assert int(cross.sum())==len(s['cue_positions'])*len(s['context_positions'])
            records.append({'row_ids':[s[k]['row_id'] for k in ('context0','context1')],
                            'causal_edges_per_head':n*(n+1)//2,'single_cue_edges_per_head':int(single.sum()),
                            'mixed_edges_per_head':int(cross.sum()),'mixed_query_positions':cross.any(-1).nonzero().flatten().tolist(),
                            'semantic_query_has_possible_mixed_read':bool(cross[-1].any())})
        panels[panel]={'squares':len(records),'causal_edges_per_head':sum(r['causal_edges_per_head'] for r in records),
                       'single_cue_edges_per_head':sum(r['single_cue_edges_per_head'] for r in records),
                       'mixed_edges_per_head':sum(r['mixed_edges_per_head'] for r in records),
                       'semantic_queries_with_possible_mixed_read':sum(r['semantic_query_has_possible_mixed_read'] for r in records),
                       'records':records}
    result={'passed':True,'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'support_code_sha256':hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest(),
            'panels':panels,'trained_forwards':0,'selection_uses_model_outcomes':False,
            'scope':'Structural support for FIRST token-local attention, including causal mask. Possible edges are not causal importance or identified task circuits; edge counts are not whole-model savings.'}
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({p:{k:v for k,v in r.items() if k!='records'} for p,r in panels.items()}))


if __name__=='__main__':main()
