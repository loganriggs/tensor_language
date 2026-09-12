"""Frozen contextual holdout, retaining lexical clusters and reporting every cell."""
from pathlib import Path
import json,hashlib
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent
PREFIXES=[
 'Here is the next example. ',
 'In a quiet village near the coast, everyone knows the routine. ',
 'The supervisor wrote the following sentence on the board: ',
 'The group has been reviewing the schedule for several hours. ',
 'Without giving any further details, the narrator continues. ',
 'Note: the next statement concerns an ordinary event. ',
 'The document includes a short illustration for the reader. ',
 'After the meeting ended, the speaker gave this example. ']
def main():
 out=P/'PRODUCER_METRIC_CONTEXT_HOLDOUT_V1_ROWS.json';assert not out.exists()
 t=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
 old=json.loads((P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json').read_text())['rows'];prior={tuple(r[s+'_ids']) for r in old for s in ('base','donor')};rows=[]
 for context,prefix in enumerate(PREFIXES):
  for row in old:
   family=row['family'];lexeme=row['lexeme'];intro=f'They were asked to {lexeme}. '
   if family=='A1':base,donor=intro+'On ordinary mornings this mechanic',intro+'On ordinary mornings these mechanics'
   elif family=='past':base,donor=intro+'On weekdays they',intro+'A moment ago they'
   elif family=='progressive':base,donor=intro+'Normally they',intro+'At this moment they are'
   else:
    plural=t.decode([row['base_answer_id'] if row['group_number']%2 else row['donor_answer_id']]).strip()
    base,donor=f'Their catalog lists {plural}. I would like a single',f'Their catalog lists {plural}. I would like several'
   if family in ('A1','A2') and row['group_number']%2:base,donor=donor,base
   new=dict(family=family+':context'+str(context),task_family=family,context=context,group_number=row['group_number'],lexeme=lexeme)
   for side,text in [('base',prefix+base),('donor',prefix+donor)]:
    ids=t.encode(text).ids;answer=row[side+'_answer_id'];foil=row[side+'_foil_id']
    assert tuple(ids) not in prior and t.encode(text+t.decode([answer])).ids==ids+[answer]
    new.update({side+'_text':text,side+'_ids':ids,side+'_prediction_position':len(ids)-1,side+'_answer_id':answer,side+'_foil_id':foil})
   new['row_id']=hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest();rows.append(new)
 assert len(rows)==512 and len({r['row_id'] for r in rows})==512
 buckets={}
 for row in rows:
  for side in ('base','donor'):
   n=len(row[side+'_ids']);buckets[n]=buckets.get(n,0)+1
 r=dict(rows=rows,authority_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),prefixes=PREFIXES,
  price=dict(sequences=1024,length_buckets=buckets,maximum_length=max(buckets),body_forwards_batch8=sum((n+7)//8 for n in buckets.values())),
  scope='New surrounding contexts and core constructions; existing16lexical groups reused.32cells of16pairs;512pairs are not512independent lexical samples. No model outcomes or fitting; not corpus/pretraining OOD.')
 out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
