"""Deterministic authored regional panels; existing frozen builders remain immutable."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import tiktoken

def construct_panel(forms_by_family,pairs,endpoints):
 enc=tiktoken.get_encoding('gpt2')
 contexts=[];rows=[]
 for variant,forms in forms_by_family.items():
  for ti,form in enumerate(forms):
   for pi,pair in enumerate(pairs):
    cid=len(contexts);seqs=[]
    for city in pair:
     text=form.format(city=city);ids=enc.encode(text);tok=enc.encode(' '+city);assert len(tok)==1 and ids.count(tok[0])==1
     seqs.append((text,ids,ids.index(tok[0])))
    assert len(seqs[0][1])==len(seqs[1][1]) and seqs[0][2]==seqs[1][2] and sum(a!=b for a,b in zip(seqs[0][1],seqs[1][1]))==1
    contexts.append({'context_id':cid,'variant':variant,'template':ti,'pair':pi,'template_text':form,'panel_status':'fresh at freeze'})
    for ei,(uk,us) in enumerate(endpoints):
     ui=enc.encode(' '+uk);si=enc.encode(' '+us);assert len(ui)==len(si)==1
     for cue,city,(text,ids,cp) in zip(['British','American'],pair,seqs):
      rows.append({'row_id':len(rows),'context_id':cid,'variant':variant,'template':ti,'pair':pi,'cue':cue,'city':city,'text':text,'ids':ids,'city_position':cp,'destination_positions':list(range(cp+1,len(ids)-1)),'endpoint':ei,'uk_id':ui[0],'us_id':si[0]})
 return contexts,rows

def freeze_panel(root,stem,forms,pairs,endpoints,selection):
 root=Path(root);out=root/(stem+'_ROWS.json');assert not out.exists()
 contexts,rows=construct_panel(forms,pairs,endpoints);seqs={tuple(r['ids']) for r in rows};assert len(seqs)==2*len(contexts)
 old=set()
 for f in root.glob('*_ROWS.json'):
  doc=json.loads(f.read_text())
  if isinstance(doc,dict):
   for row in doc.get('rows',[]):
    if isinstance(row,dict) and isinstance(row.get('ids'),list):old.add(tuple(row['ids']))
 assert not seqs&old,'Exact prompt collision with prior panel'
 doc={'schema':'regional.prospective.rows.v1','utc':datetime.now(timezone.utc).isoformat(),'contexts':contexts,'rows':rows,'variants':list(forms),'endpoints':endpoints,'selection':selection}
 out.write_text(json.dumps(doc,indent=2)+'\n')
 receipt={'pred_a':True,'rows':len(rows),'contexts':len(contexts),'token_sequences':len(seqs),'fresh_variant_prior_prefix_overlap':0,'prior_prefix_count':len(old),'row_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (root/(stem+'_CPU_CONTROL.json')).write_text(json.dumps(receipt,indent=2)+'\n');return receipt
