"""Freeze grammatical contrasts before native outcomes; no corpus or factor fit."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    root=Path(__file__).parent
    output=root/'FROZEN_BRANCH_MORPHOLOGY_V1_ROWS.json'
    assert not output.exists()
    tok=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    old=json.loads((root/'FROZEN_BRANCH_TENSE_V6_ROWS.json').read_text())['rows']
    verbs=['run','walk','work','play','sleep','sing','read','talk','laugh','wait','jump','cook','drive','write','dance','help']
    nouns=['cat','dog','book','car','tree','chair','bird','house','ball','cup','boat','shoe','coin','apple','flower','bottle']
    def tid(word):
        ids=tok.encode(' '+word,add_special_tokens=False).ids
        assert len(ids)==1,(word,ids)
        assert tok.decode(ids)==' '+word
        return ids[0]
    rows=[]
    for i,(verb,noun) in enumerate(zip(verbs,nouns)):
        suff=verb+'s'
        intro=f'They like to {verb}. Every day '
        templates=[('A1',intro+'he',intro+'they',suff,verb),
                   ('A2',f'We are counting {noun}s. I counted exactly one',f'We are counting {noun}s. I counted exactly two',noun,noun+'s'),
                   ('P',intro+('he' if i%2==0 else 'they'),intro+('she' if i%2==0 else 'we'),suff if i%2==0 else verb,suff if i%2==0 else verb)]
        for family,base,donor,ba,da in templates:
            if family!='P' and i%2:
                base,donor,ba,da=donor,base,da,ba
            forms=(noun,noun+'s') if family=='A2' else (verb,suff)
            bf=next(w for w in forms if w!=ba);df=next(w for w in forms if w!=da)
            row=dict(family=family,group_number=i,base_text=base,donor_text=donor,
                     direction='base_to_suffix' if da==forms[1] else 'suffix_to_base',
                     construction_checks=dict(single_token_answers=True,final_position=True,matched_lexeme=True))
            for side,text,answer,foil in [('base',base,ba,bf),('donor',donor,da,df)]:
                ids=tok.encode(text,add_special_tokens=False).ids
                row.update({side+'_ids':ids,side+'_prediction_position':len(ids)-1,
                            side+'_answer_id':tid(answer),side+'_foil_id':tid(foil)})
            row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
            rows.append(row)
        control=dict(old[4*i+3]);assert control['family']=='C'
        control['direction']='base_to_suffix' if i%2==0 else 'suffix_to_base'
        rows.append(control)
    sequences=[r[s+'_ids'] for r in rows for s in ('base','donor')]
    counts={length:sum(len(s)==length for s in sequences) for length in sorted(set(map(len,sequences)))}
    physical_n=min(8,counts[min(counts)])
    price=dict(body_forwards=sum((n+7)//8 for n in counts.values())+2,
               sequences=len(sequences)+2*physical_n,length_range=[min(counts),max(counts)])
    encoded=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
    result=dict(rows=rows,authority_sha256=hashlib.sha256(encoded).hexdigest(),price=price,
        scope='Frozen spelling-guided subject agreement and count noun screen. Historical V6 unrelated controls reused. '
              'Not new grammar discovery, training-disjoint or OOD evaluation; factors unchanged.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(authority_sha256=result['authority_sha256'],price=price,length_buckets=counts)))


if __name__=='__main__':main()
