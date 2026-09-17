"""Freeze genuinely new authored constructions for already frozen predictors."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import tiktoken
P=Path(__file__).resolve().parent;STEM='CITY_REMOVAL_ENDPOINT_FRESH_V1'
FORMS={'word_index': ['Word index prepared in {city}: "The document includes the word', 'The {city} word index contains this line: "The document includes the word'], 'exact_copy': ['Copy the line printed in {city}: "The document includes the word', 'Reproduce the wording used by the editor in {city}: "The document includes the word'], 'plain_word_note': ['A publication from {city} discusses local vocabulary. The document includes the word', 'An editor working in {city} describes a manuscript. The document includes the word'], 'line_word_note': ['Vocabulary record\nPlace: {city}\nThe document includes the word', 'Editorial notes\nFiled in {city}\nText\nThe document includes the word'], 'reported_word_note': ['The reviewer in {city} said that the document includes the word', 'A proofreader based in {city} noted that the document includes the word']}

def main():
 out=P/(STEM+'_ROWS.json');assert not out.exists();enc=tiktoken.get_encoding('gpt2')
 prior=json.loads((P/'TYPED_FACE_STRUCTURE_V1_ROWS.json').read_text());endpoints=[('colour','color'),('centre','center'),('favourite','favorite'),('flavour','flavor'),('travelling','traveling'),('cancelled','canceled')];pairs=[('York','Portland'),('Oxford','Seattle')];assert not {tuple(e) for e in endpoints}&{tuple(e) for e in prior['endpoints']}
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
 out.write_text(json.dumps({'schema':'regional.prospective.rows.v1','utc':datetime.now(timezone.utc).isoformat(),'contexts':contexts,'rows':rows,'variants':list(FORMS),'endpoints':endpoints,'selection':'Ten authored constructions, frozen without model execution. Cities held out from donor-free removal selection panel, six endpoints held out from removal/normalization selection; no prior token-prefix overlap. Frozen four-feature baseline comparison planned; no globally unseen-city claim.'},indent=2)+'\n')
 control={'pred_a':True,'rows':len(rows),'contexts':len(contexts),'token_sequences':40,'fresh_variant_prior_prefix_overlap':len(seqs&old),'prior_prefix_count':len(old),'row_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 (P/(STEM+'_CPU_CONTROL.json')).write_text(json.dumps(control,indent=2)+'\n');print(json.dumps(control))
if __name__=='__main__':main()
