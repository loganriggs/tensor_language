"""Outcome-blind structural controls, with explicit opened/fresh labels."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import tiktoken
P=Path(__file__).resolve().parent
STEM='TYPED_FACE_STRUCTURE_V1'
def main():
 out=P/(STEM+'_ROWS.json');assert not out.exists()
 prior=json.loads((P/'ODD_FRAMING_FRESH_V1_ROWS.json').read_text());enc=tiktoken.get_encoding('gpt2')
 bases=prior['templates'];pairs=prior['city_pairs'];endpoints=prior['endpoints']
 variants=['base','no_colon','no_quote','neither','short_frame']
 rows=[];contexts=[]
 for variant in variants:
  for ti,base in enumerate(bases):
   form=base
   if variant in ['no_colon','neither']:form=form.replace(':','')
   if variant in ['no_quote','neither']:form=form.replace('"','')
   if variant=='short_frame':form=['A letter from {city}: "We invited our new','A recording from {city}: "We invited our new'][ti]
   for pi,pair in enumerate(pairs):
    cid=len(contexts);seqs=[]
    for city in pair:
     text=form.format(city=city);ids=enc.encode(text);token=enc.encode(' '+city);assert len(token)==1 and ids.count(token[0])==1
     cp=ids.index(token[0]);seqs.append((text,ids,cp))
    assert len(seqs[0][1])==len(seqs[1][1]) and seqs[0][2]==seqs[1][2]
    assert sum(a!=b for a,b in zip(seqs[0][1],seqs[1][1]))==1
    contexts.append({'context_id':cid,'variant':variant,'template':ti,'pair':pi,'template_text':form,'panel_status':'opened template reused' if variant=='base' else 'fresh structural variant at freeze'})
    for ei,(uk,us) in enumerate(endpoints):
     ui=enc.encode(' '+uk);si=enc.encode(' '+us);assert len(ui)==len(si)==1
     for cue,city,(text,ids,cp) in zip(['British','American'],pair,seqs):
      rows.append({'row_id':len(rows),'context_id':cid,'variant':variant,'template':ti,'pair':pi,'cue':cue,'city':city,'text':text,'ids':ids,'city_position':cp,'destination_positions':list(range(cp+1,len(ids)-1)),'endpoint':ei,'uk_id':ui[0],'us_id':si[0]})
 # Compare modified prefixes against every prior authored regional row manifest.
 prior_sequences=set()
 for file in P.glob('ODD_*_ROWS.json'):
  doc=json.loads(file.read_text())
  for r in doc.get('rows',[]):
   if isinstance(r.get('ids'),list):prior_sequences.add(tuple(r['ids']))
 novel={tuple(r['ids']) for r in rows if r['variant']!='base'}
 overlap=len(novel&prior_sequences);assert overlap==0
 doc={'schema':'regional.structure.rows.v1','utc':datetime.now(timezone.utc).isoformat(),'contexts':contexts,'rows':rows,'variants':variants,'endpoints':endpoints,'selection':'Authored structure changes frozen without model execution. Base templates reused. Short-frame changes both content and clause position; not a pure position intervention.'}
 out.write_text(json.dumps(doc,indent=2)+'\n')
 control={'pred_a':True,'rows':len(rows),'contexts':len(contexts),'token_sequences':len({tuple(r['ids']) for r in rows}),'fresh_variant_prior_prefix_overlap':overlap,'prior_prefix_count':len(prior_sequences),'row_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (P/(STEM+'_CPU_CONTROL.json')).write_text(json.dumps(control,indent=2)+'\n');print(json.dumps(control))
if __name__=='__main__':main()
