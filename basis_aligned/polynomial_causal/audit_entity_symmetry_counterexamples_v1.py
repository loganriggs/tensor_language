"""Complete saved-pair failure corpus and additive JS sample accounting."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import torch
from entity_equivariance_reference import information_radius

BASE=Path(__file__).resolve().parent;OUT=BASE/'ENTITY_SYMMETRY_COUNTEREXAMPLES_V1.json'
ROWS=BASE/'GLOBAL_ENTITY_GATE_REUSE_V1_ROWS.pt'


def main():
    torch.set_num_threads(2);assert not OUT.exists();data=torch.load(ROWS,map_location='cpu',weights_only=True);results={};records=[]
    for pop,block in data.items():
        categories=defaultdict(list);old=block['query_logits']['native'];new=block['query_logits']['renamed_native'];n=len(old)
        for i,m in enumerate(block['metadata']):
            mapping=torch.arange(29);mapping[block['tokens'][i,:48:2]]=block['renamed_tokens'][i,:48:2]
            aligned=new[i,mapping];p=old[i].softmax(-1);q=aligned.softmax(-1);a=int(p.argmax());b=int(q.argmax());answer=m['answer']
            category=('both_correct' if a==answer and b==answer else 'correctness_changed' if (a==answer)!=(b==answer)
                      else 'both_wrong_same' if a==b else 'both_wrong_different')
            js=float(information_radius(old[i],aligned)[0].clamp_min(0));categories[category].append(js)
            records.append({'population':pop,'row':i,'world':m['world'],'query':m['query'],'hop':m['hop'],'category':category,'js':js,
                            'correct_answer_original_labels':answer,'old_prediction':a,'renamed_prediction_original_labels':b,
                            'old_gold_probability':float(p[answer]),'renamed_gold_probability':float(q[answer]),
                            'entity_permutation':mapping[:24].tolist(),'old_tokens':block['tokens'][i].tolist(),'renamed_tokens':block['renamed_tokens'][i].tolist()})
        results[pop]={k:{'pairs':len(v),'mean_js_within_category':sum(v)/len(v),'contribution_to_population_mean_js':sum(v)/n} for k,v in categories.items()}
    disagreements=sorted((r for r in records if r['old_prediction']!=r['renamed_prediction_original_labels']),key=lambda r:(-r['js'],r['population'],r['row']))
    receipt={'scope':'opened discovery corpus, all1536 pairs retained; ranking is not a fresh evaluation or a filter on the lower bound',
             'populations':results,'all_pairs':records,'highest_js_disagreement_case_ids':[{'population':r['population'],'row':r['row']} for r in disagreements[:16]],
             'input_rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'populations':results,'disagreement_pairs':len(disagreements),'top_cases':receipt['highest_js_disagreement_case_ids']},indent=2))


if __name__=='__main__':main()
