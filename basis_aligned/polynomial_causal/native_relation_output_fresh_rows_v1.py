"""New lexical/construction validation rows for immutable output branches."""
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer


def main():
    p=Path(__file__).parent;out=p/'NATIVE_RELATION_OUTPUT_FRESH_V1_ROWS.json';assert not out.exists()
    t=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
    old=set()
    for stem in ('FROZEN_BRANCH_MORPHOLOGY_V1','NATIVE_RELATION_HOLDOUT_V1','NATIVE_RELATION_NEIGHBOR_V1'):
        for r in json.loads((p/(stem+'_ROWS.json')).read_text())['rows']:
            for s in ('base','donor'):old.update([r[s+'_answer_id'],r[s+'_foil_id']])
    verbs=[line.split() for line in ['push pushes pushed pushing','pull pulls pulled pulling','kick kicks kicked kicking',
        'touch touches touched touching','fix fixes fixed fixing','carry carries carried carrying','enjoy enjoys enjoyed enjoying',
        'finish finishes finished finishing','start starts started starting','stop stops stopped stopping','turn turns turned turning',
        'move moves moved moving','change changes changed changing','look looks looked looking','love loves loved loving','check checks checked checking']]
    nouns=[line.split() for line in ['bed beds','desk desks','room rooms','wall walls','road roads','shop shops','star stars','beach beaches',
        'class classes','bus buses','horse horses','stone stones','key keys','toy toys','map maps','door doors']]
    def token(word):
        ids=t.encode(' '+word).ids
        assert len(ids)==1 and t.decode(ids)==' '+word and ids[0] not in old,word
        return ids[0]
    rows=[]
    for i,((verb,s,past,ing),(noun,plural)) in enumerate(zip(verbs,nouns)):
        intro=f'They plan to {verb}. '
        specs=[('A1',intro+'After lunch the guide',intro+'After lunch the guides',s,verb),
            ('A2',f'They sell {plural} here. She bought one',f'They sell {plural} here. She bought many',noun,plural),
            ('past',intro+'Each week they',intro+'Last week they',verb,past),
            ('progressive',intro+'They usually',intro+'They have been',verb,ing)]
        for family,base,donor,ba,da in specs:
            forms=(ba,da);direction='base_to_inflected'
            if family in ('A1','A2'):
                if i%2:base,donor,ba,da=donor,base,da,ba
                direction='base_to_suffix' if da in (s,plural) else 'suffix_to_base'
            row=dict(family=family,direction=direction,group_number=i,lexeme=noun if family=='A2' else verb,
                base_text=base,donor_text=donor,answer_ids_disjoint=True)
            for side,text,answer in [('base',base,ba),('donor',donor,da)]:
                foil=next(x for x in forms if x!=answer);ids=t.encode(text).ids
                row.update({side+'_ids':ids,side+'_prediction_position':len(ids)-1,side+'_answer_id':token(answer),side+'_foil_id':token(foil)})
            row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();rows.append(row)
    sequences=[r[s+'_ids'] for r in rows for s in ('base','donor')]
    buckets={n:sum(len(s)==n for s in sequences) for n in sorted(set(map(len,sequences)))}
    first=min(8,buckets[min(buckets)])
    result=dict(rows=rows,authority_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        price=dict(body_forwards=sum((n+7)//8 for n in buckets.values())+2,sequences=128+2*first,length_range=[min(buckets),max(buckets)],tail_rows=704),
        scope='All answer tokens new to previous behavioral panels; same verbs across task/control families. No outcome filtering, fitting or pretraining-disjointness claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
