"""CPU post-screen accounting, with no refit or threshold changes."""
import hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=Path(__file__).resolve().parent;src=p/'UNEMBEDDING_QUADRATIC_READER_GEOMETRY_V1_RESULT.json'
    r=json.loads(src.read_text()); rows=r['rows']; assert len(rows)==518
    candidates=[x for x in rows if x['candidate']]
    assert len(candidates)==r['candidate_count']
    unique=sorted({tuple(sorted((x['token_id'],x['partner_id']))) for x in candidates})
    summary={}
    for key in ('unembedding','quadratic','trace_free'):
        errors=np.array([x[key]['relative_error'] for x in rows])
        centroid=np.array(r['centroid_relative_errors'][key])
        summary[key]={'nearest_trace_free_partner_error_quantiles':np.quantile(errors,[0,.1,.5,.9,1]).tolist(),
                      'fixed_centroid_error_quantiles':np.quantile(centroid,[0,.1,.5,.9,1]).tolist()}
    same=sum(x['leaf']==x['partner_leaf'] for x in rows)
    counts={str(leaf):sum(x['leaf']==leaf for x in rows) for leaf in range(16)}
    expected=sum(n*(n-1) for n in counts.values())/(len(rows)*(len(rows)-1))
    exemplars=sorted(rows,key=lambda x:(x['trace_free']['relative_error'],x['token_id']))[:12]
    # Decode only from the existing local tokenizer cache, if present.
    decoder_status='not available in local cache'
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('gpt2',local_files_only=True)
        exemplars=[dict(x,token_text=tokenizer.decode([x['token_id']]),partner_text=tokenizer.decode([x['partner_id']])) for x in exemplars]
        decoder_status='gpt2 local cache'
    except (OSError,ImportError):pass
    out={'schema':'unembedding.quadratic_reader_geometry.audit.v1',
         'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'candidate_reader_count':len(candidates),'unique_candidate_pairs':unique,
         'error_quantiles_order':[0,.1,.5,.9,1],'geometry':summary,
         'same_leaf_nearest_fraction':same/len(rows),'same_leaf_random_partner_expectation':expected,
         'sample_leaf_counts':counts,'closest_trace_free_exemplars':exemplars,
         'decoder_status':decoder_status,
         'scope':'Descriptive accounting of frozen weight screen; nearest partners selected on weights, no independent validation. Frobenius similarity is not pointwise relative response or behavioral equivalence.'}
    with (p/'UNEMBEDDING_QUADRATIC_READER_GEOMETRY_AUDIT_V1_RESULT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:out[k] for k in ('candidate_reader_count','unique_candidate_pairs','geometry','same_leaf_nearest_fraction','same_leaf_random_partner_expectation','decoder_status')},indent=2))

if __name__=='__main__':main()
