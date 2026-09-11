"""Freeze neighboring inflection controls before accessing native outcomes."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=Path(__file__).parent;output=p/'NATIVE_RELATION_NEIGHBOR_V1_ROWS.json'
    assert not output.exists()
    tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    verbs='walk work play talk laugh wait jump cook help clean paint watch open close listen learn'.split()
    rows=[]
    def token(word):
        ids=tokenizer.encode(' '+word).ids
        assert len(ids)==1 and tokenizer.decode(ids)==' '+word,word
        return ids[0]
    for family in ('past','progressive'):
        for verb in verbs:
            form=(verb+'d' if verb.endswith('e') else verb+'ed') if family=='past' else (verb[:-1]+'ing' if verb.endswith('e') else verb+'ing')
            intro=f'They like to {verb}. '
            base=intro+('Every day they' if family=='past' else 'They often')
            donor=intro+('Yesterday they' if family=='past' else 'They are currently')
            row=dict(family=family,verb=verb,form=form,base_text=base,donor_text=donor)
            for side,text,answer,foil in [('base',base,verb,form),('donor',donor,form,verb)]:
                ids=tokenizer.encode(text).ids
                row.update({side+'_ids':ids,side+'_prediction_position':len(ids)-1,
                            side+'_answer_id':token(answer),side+'_foil_id':token(foil)})
            row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();rows.append(row)
    sequences=[r[s+'_ids'] for r in rows for s in ('base','donor')]
    buckets={n:sum(len(s)==n for s in sequences) for n in sorted(set(map(len,sequences)))}
    first=min(8,buckets[min(buckets)])
    result=dict(rows=rows,authority_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                price=dict(body_forwards=sum((n+7)//8 for n in buckets.values())+2,sequences=64+2*first,
                           length_range=[min(buckets),max(buckets)],tail_rows=320),
                scope='Plural-subject past and progressive controls; same lexemes across families; no outcome filtering or fitting.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
