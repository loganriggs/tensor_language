"""Fixed fresh lexical/template confirmation panel, no model outcomes used."""
from pathlib import Path
import json,tiktoken
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
PAIRS=[(' centre',' center','the town'),(' honour',' honor','a matter of personal'),(' licence',' license','a driving'),(' travelled',' traveled','how far the visitors had'),(' cancelled',' canceled','whether the flight had been'),(' grey',' gray','a shade of')]
def main():
 enc=tiktoken.get_encoding('gpt2');rows=[]
 old=json.loads((P/'REGIONAL_COMPETING_CUES_V1_ROWS.json').read_text())['rows'];old_endpoints={(r['uk_id'],r['us_id']) for r in old}
 for family in range(2):
  for pair_id,cities in enumerate((('Bristol','Boston'),('Oxford','Austin'))):
   for concept,(uk,us,clause) in enumerate(PAIRS):
    endpoints=[enc.encode(x) for x in (uk,us)];assert all(len(x)==1 for x in endpoints)
    uk_id,us_id=[x[0] for x in endpoints];assert (uk_id,us_id) not in old_endpoints
    for city,cue in zip(cities,('British','American')):
     if family==0:text=f'The copy editor is based in {city}. For the local newspaper, the editor wrote about {clause}'
     else:text=f'After moving to {city}, the journalist adopted the local spelling. The latest article discussed {clause}'
     rows.append(dict(row_id=len(rows),family=family,city_pair=pair_id,concept=concept,cue=cue,city=city,text=text,ids=enc.encode(text),uk_id=uk_id,us_id=us_id,control_ids=[670,3946],newline_control_ids=[198,11]))
 checks=validate(rows);assert len(rows)==48
 out=P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json';assert not out.exists()
 out.write_text(json.dumps(dict(rows=rows,checks=checks,scope='Fixed48new-city/new-spelling/new-template rows. Tokenization-only validation; native capability and model effects not evaluated.'),indent=2)+'\n')
 print(json.dumps(dict(rows=len(rows),min_tokens=min(len(r['ids']) for r in rows),max_tokens=max(len(r['ids']) for r in rows),checks=checks)))
if __name__=='__main__':main()
