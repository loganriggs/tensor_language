"""Saved-output diagnosis only; no model, fitting, filtering, or promotion."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch

BASE=Path(__file__).resolve().parent
ROWS=BASE/'JOIN_ORIGIN_FIELD_SWAP_V1_ROWS.pt'
OUT=BASE/'JOIN_ORIGIN_FIELD_SWAP_FAILURE_AUDIT_V1.json'


def main():
    torch.set_num_threads(2)
    assert not OUT.exists()
    data=torch.load(ROWS,map_location='cpu',weights_only=True)
    result={}
    for pop,block in data.items():
        meta=block['metadata'];logits=block['query_logits'];groups={}
        for query in range(2):
            indices=[i for i,m in enumerate(meta) if m['query']==query and m['hop']==3]
            single='swap_b_cut_a' if query==0 else 'swap_a_cut_b'
            arms={};paired=Counter();details=[]
            for arm in ('native','both_swap',single):
                categories=Counter();ranks=[];desired_p=[];old_p=[];margins=[]
                for i in indices:
                    m=meta[i];v=logits[arm][i];p=v.softmax(-1);winner=int(v.argmax())
                    chain={n['answer'] for n in meta if n['world']==m['world']}
                    category=('desired' if winner==m['desired_answer'] else 'original' if winner==m['answer']
                              else 'other_chain_entity' if winner in chain else 'unrelated')
                    categories[category]+=1
                    ranks.append(1+int((v>v[m['desired_answer']]).sum()))
                    desired_p.append(float(p[m['desired_answer']]));old_p.append(float(p[m['answer']]))
                    other=v.clone();other[m['desired_answer']]=-torch.inf
                    margins.append(float(v[m['desired_answer']]-other.max()))
                arms[arm]={'winner_counts':dict(categories),'desired_rank_counts':dict(Counter(ranks)),
                           'mean_desired_probability':sum(desired_p)/len(indices),
                           'mean_original_probability':sum(old_p)/len(indices),
                           'mean_desired_vs_best_other_logit_margin':sum(margins)/len(indices)}
            for i in indices:
                m=meta[i];j=int(logits['both_swap'][i].argmax())==m['desired_answer'];s=int(logits[single][i].argmax())==m['desired_answer']
                paired[f'joint_{j}_single_{s}']+=1
                details.append({'world':m['world'],'native_correct':int(logits['native'][i].argmax())==m['answer'],
                                'joint_correct':j,'single_correct':s,'original_answer':m['answer'],
                                'desired_answer':m['desired_answer'],'joint_winner':int(logits['both_swap'][i].argmax())})
            groups[str(query)]={'n':len(indices),'arms':arms,'paired_outcome_counts':dict(paired),'rows':details}
        result[pop]=groups
    receipt={'scope':'opened-case diagnostic; all rows retained; no model access, fit, or identification claim',
             'rows_sha256':hashlib.sha256(ROWS.read_bytes()).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'populations':result}
    OUT.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({pop:{q:{k:v for k,v in g.items() if k!='rows'} for q,g in groups.items()} for pop,groups in result.items()},indent=2))


if __name__=='__main__':main()
