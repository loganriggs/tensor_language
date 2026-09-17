"""Freeze genuinely new authored constructions for already frozen predictors."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import tiktoken
P=Path(__file__).resolve().parent;STEM='TYPED_FACE_PROSPECTIVE_V1'
FORMS={
 'archive_card':['Archive card: {city}. The writer\'s original words follow: "We invited our new','Location recorded in the archive: {city}. Copy this fragment without changes: "We invited our new'],
 'instruction_first':['Keep the original wording of this note from {city}: "We invited our new','Transcribe without editing the letter sent from {city}: "We invited our new'],
 'unquoted_prose':['A newspaper in {city} printed the following sentence. We invited our new','The diary found in {city} contained these words. We invited our new'],
 'line_break':['Correspondence from {city}\nThe final line reads\nWe invited our new','Interview recorded in {city}\nOriginal transcript\nWe invited our new'],
 'indirect_report':['The librarian in {city} read from the manuscript that we invited our new','A resident of {city} wrote in the guest book that we invited our new']}
def main():
 out=P/(STEM+'_ROWS.json');assert not out.exists();enc=tiktoken.get_encoding('gpt2')
 prior=json.loads((P/'TYPED_FACE_STRUCTURE_V1_ROWS.json').read_text());endpoints=prior['endpoints'];pairs=[('Cambridge','Phoenix'),('Leeds','Chicago')]
 old=set()
 for f in P.glob('*_ROWS.json'):
  doc=json.loads(f.read_text())
  if not isinstance(doc,dict):continue
  for row in doc.get('rows',[]):
   if isinstance(row,dict) and isinstance(row.get('ids'),list):old.add(tuple(row['ids']))
 contexts=[];rows=[]
 for variant,forms in FORMS.items():
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
 seqs={tuple(r['ids']) for r in rows};assert not seqs&old and len(seqs)==40
 out.write_text(json.dumps({'schema':'regional.prospective.rows.v1','utc':datetime.now(timezone.utc).isoformat(),'contexts':contexts,'rows':rows,'variants':list(FORMS),'endpoints':endpoints,'selection':'Ten authored constructions, frozen without model execution. Same named city pairs and endpoint meanings as fixed baseline training; no prior token-prefix overlap.'},indent=2)+'\n')
 control={'pred_a':True,'rows':len(rows),'contexts':len(contexts),'token_sequences':40,'fresh_variant_prior_prefix_overlap':len(seqs&old),'prior_prefix_count':len(old),'row_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (P/(STEM+'_CPU_CONTROL.json')).write_text(json.dumps(control,indent=2)+'\n');print(json.dumps(control))
if __name__=='__main__':main()
