"""Prospective fixed rows, no model scores or candidate result access."""
import json,hashlib,random
from pathlib import Path
import torch,tiktoken
from regional_cue_row_check_v1 import validate
from scalar_producers_context_transfer_rows_v1 import PAIRS
P=Path(__file__).resolve().parent

def main():
 enc=tiktoken.get_encoding('gpt2');old=json.loads((P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json').read_text());oldids={tuple(r['ids']) for r in old['rows']};rows=[]
 for family in range(3):
  for city_pair,cities in enumerate((('Glasgow','Phoenix'),('Cambridge','Detroit'))):
   for concept,(uk,us,stem) in enumerate(PAIRS):
    u,s=enc.encode(uk),enc.encode(us);assert len(u)==len(s)==1
    for cue,city in zip(('British','American'),cities):
     other=cities[1] if city==cities[0] else cities[0]
     if family==0:text=f'A lifelong resident of {city} wrote in a personal diary: "{stem}'
     elif family==1:text=f'The teacher in {city} asked for local spelling in a school essay. The student wrote: "{stem}'
     else:text=f'The writer lives in {city}. The reader lives in {other}. Using the writer\'s local spelling, the message begins: "{stem}'
     ids=enc.encode(text);assert tuple(ids) not in oldids
     rows.append(dict(row_id=len(rows),family=family,city_pair=city_pair,concept=concept,cue=cue,city=city,text=text,ids=ids,uk_id=u[0],us_id=s[0],control_ids=[670,3946]))
 # Two-role family changes two city mentions by design; apply single-cue validator
 # only to the two single-city families, and explicitly check paired role swaps below.
 checks=validate(rows[:48])
 for a,b in zip(rows[48::2],rows[49::2]):
  assert len(a['ids'])==len(b['ids']) and a['uk_id']==b['uk_id'] and a['us_id']==b['us_id']
  different=[j for j,(x,y) in enumerate(zip(a['ids'],b['ids'])) if x!=y]
  assert len(different)==2
  j,k=different;assert a['ids'][j]==b['ids'][k] and a['ids'][k]==b['ids'][j]
 prior=json.loads((P/'SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_ROWS.json').read_text());source=Path(prior['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==prior['source_sha']
 data=torch.load(source,map_location='cpu',weights_only=True);exclude={r['source_row'] for r in prior['rows'] if r['pool']=='fineweb'}|set(range(24));eligible=[i for i in range(192) if i not in exclude and bool((data[i,64:257]==198).any())];random.Random(191224).shuffle(eligible);chosen=eligible[:32];assert len(chosen)==32
 natural=[]
 for ordinal,i in enumerate(chosen):
  pos=64+int(torch.nonzero(data[i,64:257]==198)[0,0]);natural.append(dict(row_id=ordinal,family=ordinal//16,pool='fineweb',source_row=i,target_position=pos,ids=data[i,:pos].tolist(),newline_id=198,comma_id=11))
 out=P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json';assert not out.exists();out.write_text(json.dumps(dict(regional=rows,natural=natural,regional_single_city_checks=checks,newline_excluded_indices=sorted(exclude),source=str(source),source_sha=prior['source_sha'],seed=191224,scope='Prospective new templates including explicit opposed-city writer/reader challenge; city/endpoints reused. New newline cache row indices, not document or corpus disjointness. No model score filtering; candidate not yet selected.'),indent=2)+'\n');print(dict(regional=len(rows),natural=len(natural),max_regional_tokens=max(len(r['ids']) for r in rows),max_natural_tokens=max(len(r['ids']) for r in natural),remaining_eligible=len(eligible)))
if __name__=='__main__':main()
