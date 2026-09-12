"""Fresh frozen contextual panel; token-only construction and prior-panel novelty checks."""
import json,tiktoken
from pathlib import Path
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
PAIRS=[(' neighbours',' neighbors','We invited our new'),(' organise',' organize','Next week, we will'),(' realise',' realize','Only then did I'),(' labelled',' labeled','The parcels were clearly'),(' defence',' defense','The lawyer prepared the'),(' metre',' meter','The cable measured one')]
def main():
 enc=tiktoken.get_encoding('gpt2');oldcities=set();oldpairs=set();sources=[]
 for f in sorted(list(P.glob('REGIONAL*ROWS.json'))+list(P.glob('STRUCTURED_PRODUCER*ROWS.json'))):
  sources.append(f.name)
  for r in json.loads(f.read_text()).get('rows',[]):
   if 'city' in r:oldcities.add(r['city'])
   if 'uk_id' in r and 'us_id' in r:oldpairs.add((r['uk_id'],r['us_id']))
 rows=[]
 for family in range(2):
  for city_pair,cities in enumerate((('Glasgow','Phoenix'),('Cambridge','Detroit'))):
   assert not (set(cities)&oldcities)
   for concept,(uk,us,stem) in enumerate(PAIRS):
    ends=[enc.encode(x) for x in (uk,us)];assert all(len(x)==1 for x in ends);uk_id,us_id=[x[0] for x in ends];assert (uk_id,us_id) not in oldpairs
    for cue,city in zip(('British','American'),cities):
     if family==0:text=f'An email sent from {city} contained this sentence: "{stem}'
     else:text=f'A person who grew up in {city} was writing a letter. The next sentence began, "{stem}'
     rows.append(dict(row_id=len(rows),family=family,city_pair=city_pair,concept=concept,cue=cue,city=city,text=text,ids=enc.encode(text),uk_id=uk_id,us_id=us_id,control_ids=[670,3946]))
 checks=validate(rows);out=P/'SCALAR_PRODUCERS_CONTEXT_TRANSFER_V1_ROWS.json';assert not out.exists()
 out.write_text(json.dumps(dict(rows=rows,checks=checks,novelty_source_files=sources,scope='Fixed new city/spelling/template panel relative to inspected regional and producer rows. No model-score filtering or pretraining/corpus OOD claim.'),indent=2)+'\n')
 print(json.dumps(dict(checks=checks,min_tokens=min(len(r['ids']) for r in rows),max_tokens=max(len(r['ids']) for r in rows))))
if __name__=='__main__':main()
