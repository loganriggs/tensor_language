"""Outcome-blind fresh lexical/construction rows for the frozen outer32 interface."""
from pathlib import Path
import hashlib,json
from tokenizers import Tokenizer
P=Path(__file__).resolve().parent
VERBS='''paint paints painted painting
train trains trained training
clean cleans cleaned cleaning
jump jumps jumped jumping
cook cooks cooked cooking
walk walks walked walking
talk talks talked talking
play plays played playing
work works worked working
help helps helped helping
call calls called calling
need needs needed needing
want wants wanted wanting
open opens opened opening
close closes closed closing
wash washes washed washing
watch watches watched watching
visit visits visited visiting
listen listens listened listening
wait waits waited waiting
smile smiles smiled smiling
laugh laughs laughed laughing
rest rests rested resting
dance dances danced dancing
hunt hunts hunted hunting
burn burns burned burning
lift lifts lifted lifting
return returns returned returning
follow follows followed following
learn learns learned learning
collect collects collected collecting
deliver delivers delivered delivering
repair repairs repaired repairing
travel travels traveled traveling
order orders ordered ordering
offer offers offered offering
accept accepts accepted accepting
print prints printed printing
count counts counted counting
shout shouts shouted shouting'''
NOUNS='''planet planets
river rivers
island islands
window windows
garden gardens
car cars
book books
dog dogs
cat cats
tree trees
house houses
table tables
chair chairs
phone phones
train trains
boat boats
plane planes
truck trucks
flower flowers
coin coins
ticket tickets
bottle bottles
cup cups
plate plates
shirt shirts
shoe shoes
hat hats
bag bags
box boxes
gift gifts
letter letters
picture pictures
camera cameras
computer computers
bridge bridges
tower towers
park parks
lake lakes
apple apples
egg eggs
wheel wheels
ship ships
engine engines
button buttons
screen screens
cloud clouds
mountain mountains
valley valleys
forest forests
branch branches
seed seeds
leaf leaves
mirror mirrors
blanket blankets
pillow pillows
candle candles'''

def main():
 out=P/'QUARTIC_OUTER32_FRESH_V1_ROWS.json';assert not out.exists()
 tokenizer=Tokenizer.from_file('/workspace/.hf_home/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json')
 old_tokens=set();old_texts=set();authorities={}
 for stem in ('FROZEN_BRANCH_MORPHOLOGY_V1','NATIVE_RELATION_HOLDOUT_V1','NATIVE_RELATION_NEIGHBOR_V1','NATIVE_RELATION_OUTPUT_FRESH_V1'):
  path=P/(stem+'_ROWS.json');authorities[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  for row in json.loads(path.read_text())['rows']:
   for side in ('base','donor'):
    old_tokens.update([row[side+'_answer_id'],row[side+'_foil_id']])
    old_texts.add(tuple(row[side+'_ids']))
 rejected=[]
 def eligible(lines):
  accepted=[]
  for line in lines.splitlines():
   words=line.split();ids=[tokenizer.encode(' '+word).ids for word in words]
   if any(len(a)!=1 for a in ids):rejected.append(dict(words=words,reason='multi_token'));continue
   if any(a[0] in old_tokens for a in ids):rejected.append(dict(words=words,reason='prior_answer_token'));continue
   accepted.append(words)
  return accepted
 verbs=eligible(VERBS);nouns=eligible(NOUNS)
 assert min(len(verbs),len(nouns))>=16,(len(verbs),len(nouns))
 rows=[]
 for i,((v,s,past,ing),(noun,plural)) in enumerate(zip(verbs[:16],nouns[:16])):
  intro=f'The instructions say to {v}. '
  specs=[('A1',intro+'On most days that worker',intro+'On most days those workers',s,v),
   ('A2',f'The inventory includes {plural}. She requested exactly one',f'The inventory includes {plural}. She requested several',noun,plural),
   ('past',intro+'Every morning they',intro+'Yesterday morning they',v,past),
   ('progressive',intro+'Most days they',intro+'Right now they are',v,ing)]
  for family,base,donor,ba,da in specs:
   if family in ('A1','A2') and i%2:base,donor,ba,da=donor,base,da,ba
   row=dict(family=family,group_number=i,lexeme=noun if family=='A2' else v,base_text=base,donor_text=donor)
   for side,text,answer,foil in [('base',base,ba,da),('donor',donor,da,ba)]:
    ids=tokenizer.encode(text).ids;aid=tokenizer.encode(' '+answer).ids;fid=tokenizer.encode(' '+foil).ids
    assert len(aid)==len(fid)==1 and aid[0] not in old_tokens and fid[0] not in old_tokens
    assert tuple(ids) not in old_texts
    assert tokenizer.encode(text+' '+answer).ids==ids+aid
    row.update({side+'_ids':ids,side+'_prediction_position':len(ids)-1,side+'_answer_id':aid[0],side+'_foil_id':fid[0]})
   row['row_id']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();rows.append(row)
 buckets={}
 for row in rows:
  for side in ('base','donor'):
   n=len(row[side+'_ids']);buckets[n]=buckets.get(n,0)+1
 result=dict(rows=rows,authority_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
  source_authorities=authorities,eligible_verb_count=len(verbs),eligible_noun_count=len(nouns),rejected=rejected,
  price=dict(sequences=128,length_buckets=buckets,body_forwards_batch8=sum((n+7)//8 for n in buckets.values())),
  scope='Answer tokens disjoint from four earlier registered panels; changed sentence construction. First16 tokenization-eligible lexical groups, no model outcomes or fitting. Not pretraining-disjoint or corpus OOD.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','rejected')},indent=2))
if __name__=='__main__':main()
