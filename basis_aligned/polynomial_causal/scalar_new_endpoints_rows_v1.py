"""Frozen novel cue-channel panel; tokenize/check only, no model scores."""
from pathlib import Path
import json,tiktoken
PAIRS=[(' programme',' program','The evening television'),(' humour',' humor','Everyone admired her dry'),(' catalogue',' catalog','The shop mailed its latest'),(' apologise',' apologize','For the mistake, I must'),(' offence',' offense','The remark caused serious'),(' organising',' organizing','They are busy')]
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent

def main():
 enc=tiktoken.get_encoding('gpt2');oldids=set();oldendpoints=set();sources=[]
 for f in sorted(P.glob('*ROWS.json')):
  if f.name=='SCALAR_NEW_ENDPOINTS_V1_ROWS.json':continue
  data=json.loads(f.read_text());rr=data.get('rows',data.get('regional',[])) if isinstance(data,dict) else data
  if not isinstance(rr,list):continue
  hit=False
  for r in rr:
   if not isinstance(r,dict):continue
   if 'ids' in r:oldids.add(tuple(r['ids']))
   if 'uk_id' in r and 'us_id' in r:oldendpoints.update([r['uk_id'],r['us_id']]);hit=True
  if hit:sources.append(f.name)
 citypairs=[('Leeds','Dallas'),('Birmingham','Houston')]
 rows=[]
 for family in range(3):
  for variant in range(2):
   for concept,(uk,us,stem) in enumerate(PAIRS):
    ui,si=enc.encode(uk),enc.encode(us);assert len(ui)==len(si)==1 and not set(ui+si)&oldendpoints
    for side,cue in enumerate(['British','American']):
     if family==0:word=citypairs[variant][side];text=f'The columnist has always lived in {word}. A personal message begins: "{stem}'
     elif family==1:
      word=cue;text=(f'The author is {word}. A personal note begins: "{stem}' if variant==0 else f'The {word} editor follows local spelling. The text reads: "{stem}')
     else:
      word=['UK','US'][side];text=(f'The publisher requires {word} spelling. The next sentence is: "{stem}' if variant==0 else f'According to the {word} style guide, the manuscript reads: "{stem}')
     ids=enc.encode(text);assert tuple(ids) not in oldids
     row=dict(row_id=len(rows),family=family,family_name=['city_cue','nationality','style_rule'][family],variant=variant,concept=concept,uk_token=uk,us_token=us,cue=cue,cue_word=word,text=text,ids=ids,uk_id=ui[0],us_id=si[0],control_ids=[670,3946],donor_id=len(rows)^1)
     if family==0:row['city']=word
     rows.append(row)
 checks=validate(rows);out=P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(rows=rows,checks=checks,prior_files=sources,scope='72score-free newoutputprefixes, six endpoint pairs unused in14 inspected root-level paired rowcollections. Cue templates/cities and unrelatedcontrol reused. No corpus/pretraining OOD claim, no fitting or modelscorefiltering.'),indent=2)+'\n');print(dict(rows=len(rows),pairs=len(rows)//2,checks=checks,max_tokens=max(len(r['ids']) for r in rows)))
if __name__=='__main__':main()
