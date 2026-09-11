"""Freeze fresh lexemes and grammatical constructions, no native outcomes."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=Path(__file__).parent;output=p/'NATIVE_RELATION_HOLDOUT_V1_ROWS.json';assert not output.exists()
    tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    old=json.loads((p/'FROZEN_BRANCH_MORPHOLOGY_V1_ROWS.json').read_text())['rows']
    old_answers={r[k] for r in old if r['family'] in ('A1','A2') for k in ('base_answer_id','base_foil_id')}
    verbs=[('stand','stands'),('sit','sits'),('open','opens'),('close','closes'),('arrive','arrives'),('leave','leaves'),('smile','smiles'),('cry','cries'),('listen','listens'),('learn','learns'),('teach','teaches'),('watch','watches'),('clean','cleans'),('paint','paints'),('travel','travels'),('study','studies')]
    nouns=[('table','tables'),('window','windows'),('lamp','lamps'),('roof','roofs'),('train','trains'),('bridge','bridges'),('garden','gardens'),('river','rivers'),('mountain','mountains'),('planet','planets'),('ticket','tickets'),('plate','plates'),('phone','phones'),('bag','bags'),('box','boxes'),('city','cities')]
    def token(word):
        ids=tokenizer.encode(' '+word).ids;assert len(ids)==1 and tokenizer.decode(ids)==' '+word
        assert ids[0] not in old_answers
        return ids[0]
    rows=[]
    for i,((verb,verb_s),(noun,noun_s)) in enumerate(zip(verbs,nouns)):
        relative=(i//2)%2==1
        tail=' that the managers trust' if relative else ' near the machines'
        intro=f'They often {verb} together. '
        head='workers' if i%2 else 'worker'
        answer=verb if i%2 else verb_s
        templates=[('A1',intro+'The worker'+tail,intro+'The workers'+tail,verb_s,verb,(verb,verb_s),'relative_clause' if relative else 'plural_attractor'),
            ('A2',f'They catalog {noun_s} here. The image shows a single',f'They catalog {noun_s} here. The image shows several',noun,noun_s,(noun,noun_s),'single_several'),
            ('P',intro+'The young '+head+tail,intro+'The tall '+head+tail,answer,answer,(verb,verb_s),'adjective_control')]
        for family,base,donor,ba,da,forms,construction in templates:
            if family!='P' and i%2:base,donor,ba,da=donor,base,da,ba
            bf=next(x for x in forms if x!=ba);df=next(x for x in forms if x!=da)
            row=dict(family=family,group_number=i,base_text=base,donor_text=donor,construction=construction,
                direction='base_to_suffix' if da==forms[1] else 'suffix_to_base',
                construction_checks=dict(single_token_answers=True,final_position=True,matched_lexeme=True,answer_ids_disjoint=True))
            for side,text,ans,foil in [('base',base,ba,bf),('donor',donor,da,df)]:
                ids=tokenizer.encode(text).ids
                row.update({side+'_ids':ids,side+'_prediction_position':len(ids)-1,
                            side+'_answer_id':token(ans),side+'_foil_id':token(foil)})
            row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();rows.append(row)
        control=dict(old[4*i+3]);control['construction']='historical_control';rows.append(control)
    sequences=[row[side+'_ids'] for row in rows for side in ('base','donor')]
    buckets={n:sum(len(s)==n for s in sequences) for n in sorted(set(map(len,sequences)))}
    physical=min(8,buckets[min(buckets)])
    price=dict(body_forwards=sum((n+7)//8 for n in buckets.values())+2,sequences=128+2*physical,
               length_range=[min(buckets),max(buckets)])
    result=dict(rows=rows,authority_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        price=price,scope='New validation lexemes/constructions for frozen weight program; old unrelated controls retained. '
        'No text fitting, native-outcome filtering, corpus OOD or proof of pretraining disjointness.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
