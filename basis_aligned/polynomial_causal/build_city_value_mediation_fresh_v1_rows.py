"""Outcome-blind natural-context transfer for the full-city removal operator."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,itertools,re
import tiktoken
P=Path(__file__).resolve().parent;STEM='CITY_VALUE_MEDIATION_FRESH_V1'
PAIRS=[('London','Boston'),('Bristol','Austin'),('Oxford','Chicago'),('Manchester','Seattle'),('Liverpool','Denver'),('Glasgow','Dallas'),('Edinburgh','Miami'),('Birmingham','Atlanta'),('Cambridge','Houston'),('Leeds','Phoenix')]
MAX_DOCS=20000;CELLS=20;LEFT=12;RIGHT=20

def main():
 from datasets import load_dataset
 out=P/(STEM+'_ROWS.json');assert not out.exists();enc=tiktoken.get_encoding('gpt2');lookup={}
 for pi,(uk,us) in enumerate(PAIRS):
  a,b=enc.encode(' '+uk),enc.encode(' '+us);assert len(a)==len(b)==1
  lookup[a[0]]=(pi,'British',uk,a[0],b[0]);lookup[b[0]]=(pi,'American',us,a[0],b[0])
 endpoints=json.loads((P/'CITY_FULL_FRESH_V1_ROWS.json').read_text())['endpoints'];endpoint_ids=[(enc.encode(' '+a)[0],enc.encode(' '+b)[0]) for a,b in endpoints]
 old=set();prior_documents=set()
 for f in P.glob('*_ROWS.json'):
  d=json.loads(f.read_text())
  if isinstance(d,dict):
   prior_documents.update(c['document_sha256'] for c in d.get('contexts',[]) if isinstance(c,dict) and 'document_sha256' in c)
   for r in d.get('rows',[]):
    if isinstance(r,dict) and isinstance(r.get('ids'),list):old.add(tuple(r['ids']))
 source='HuggingFaceFW/fineweb';ds=load_dataset(source,name='sample-10BT',split='train',streaming=True);contexts=[];rows=[];seen=set();scanned=0;collisions=0
 for di,doc in enumerate(itertools.islice(ds,MAX_DOCS)):
  scanned+=1;text=doc['text'];dh=hashlib.sha256(text.encode()).hexdigest()
  if dh in seen or dh in prior_documents:continue
  ids=enc.encode(text,disallowed_special=())
  for pos,tok in enumerate(ids):
   if tok not in lookup or pos<LEFT or pos+RIGHT>len(ids):continue
   before=enc.decode(ids[max(0,pos-6):pos])
   if not re.search(r'\b(?:in|from|near|outside|around|at|to)\s*$',before):continue
   window=ids[pos-LEFT:pos+RIGHT]
   if sum(t in lookup for t in window)!=1:continue
   pi,cue,city,uk,us=lookup[tok];variants={}
   for label,ct in [('British',uk),('American',us)]:
    seq=window.copy();seq[LEFT]=ct;variants[label]=seq
   if any(tuple(seq) in old for seq in variants.values()):collisions+=1;continue
   cid=len(contexts);meta=doc.get('meta',{});contexts.append({'context_id':cid,'variant':'fineweb','template':cid,'pair':pi,'source_doc_index':di,'source_token_position':pos,'document_sha256':dh,'source_meta':meta,'natural_cue':cue,'natural_text':enc.decode(window),'panel_status':'unopened for full-city operator; corpus may have prior unrelated use'})
   for ei,(ui,si) in enumerate(endpoint_ids):
    for label in ['British','American']:
     seq=variants[label];rows.append({'row_id':len(rows),'context_id':cid,'variant':'fineweb','template':cid,'pair':pi,'cue':label,'city':PAIRS[pi][0 if label=='British' else 1],'text':enc.decode(seq),'ids':seq,'city_position':LEFT,'destination_positions':list(range(LEFT+1,len(seq)-1)),'endpoint':ei,'uk_id':ui,'us_id':si,'is_untouched_natural_arm':label==cue,'document_sha256':dh})
   seen.add(dh);break
  if len(contexts)==CELLS:break
  if scanned%1000==0:print('scanned',scanned,'selected',len(contexts),flush=True)
 assert len(contexts)==CELLS,(scanned,len(contexts));assert len({tuple(r['ids']) for r in rows})==40
 doc={'schema':'regional.natural_context.rows.v1','utc':datetime.now(timezone.utc).isoformat(),'source':source,'source_split':'train','source_config':'sample-10BT','docs_scanned':scanned,'selection':'First preposition-qualified eligible occurrence in first20distinct eligible documents in stream order.32tokens,12beforecity, exactly one mapped city in window. Requires immediately preceding in/from/near/outside/around/at/to; excludes previously frozen document hashes. No model outcomes/capability filtering. One untouched arm and one city substitution. Endpoints are fixed probes, not observed next-token labels. No pretraining-disjointness guarantee.','contexts':contexts,'rows':rows,'variants':['fineweb'],'endpoints':endpoints,'city_pairs':PAIRS}
 out.write_text(json.dumps(doc,indent=2)+'\n');receipt={'pred_a':True,'contexts':len(contexts),'rows':len(rows),'sequences':40,'docs_scanned':scanned,'prior_exact_prefix_collisions_skipped':collisions,'row_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'selection_model_calls':0};(P/(STEM+'_CPU_CONTROL.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
