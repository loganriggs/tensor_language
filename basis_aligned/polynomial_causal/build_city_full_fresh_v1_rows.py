"""Unopened full-city confirmation; uses the shared deterministic panel builder."""
from pathlib import Path
import json
from regional_panel_freeze_v1 import freeze_panel,construct_panel
P=Path(__file__).resolve().parent
FORMS={
 'glossary_header':['Glossary submitted from {city}: "The writer chose the word','The {city} glossary quotes this phrase: "The writer chose the word'],
 'proof_request':['Check the exact wording of the page from {city}: "The writer chose the word','Retain the spelling in the draft received from {city}: "The writer chose the word'],
 'plain_editorial':['A correspondent based in {city} reviewed the vocabulary. The writer chose the word','The publisher in {city} discussed the draft. The writer chose the word'],
 'line_editorial':['Glossary draft\nLocation: {city}\nThe writer chose the word','Copy record\nSent from {city}\nExcerpt\nThe writer chose the word'],
 'reported_choice':['The editor from {city} explained that the writer chose the word','A language reviewer in {city} reported that the writer chose the word']}
PAIRS=[('Birmingham','Houston'),('Newcastle','Detroit')]
ENDPOINTS=[('honour','honor'),('theatre','theater'),('programme','program'),('grey','gray'),('labour','labor'),('behaviour','behavior')]
def main():
 # Verify the refactor against actual immutable rows, without rewriting them.
 import build_city_removal_endpoint_fresh_v1_rows as prior_builder
 prior=json.loads((P/'CITY_REMOVAL_ENDPOINT_FRESH_V1_ROWS.json').read_text())
 contexts,rows=construct_panel(prior_builder.FORMS,[('York','Portland'),('Oxford','Seattle')],prior['endpoints'])
 assert rows==prior['rows'] and contexts==prior['contexts']
 used={tuple(e) for stem in ['TYPED_FACE_STRUCTURE_V1','CITY_REMOVAL_ENDPOINT_FRESH_V1'] for e in json.loads((P/(stem+'_ROWS.json')).read_text())['endpoints']}
 assert not set(ENDPOINTS)&used
 r=freeze_panel(P,'CITY_FULL_FRESH_V1',FORMS,PAIRS,ENDPOINTS,'Ten new authored constructions, new cities and six endpoints held out from full-city source selection. No global unseen-city/endpoint or corpus claim. Fixed formula and opened-trained four-feature baselines; no model outcomes inspected.')
 (P/'REGIONAL_PANEL_FREEZE_V1_REPLAY_RESULT.json').write_text(json.dumps({'old_rows_exactly_reproduced':True,'rows':len(rows),'contexts':len(contexts),'old_builder_unchanged':True,'scope':'Deterministic CPU comparison to immutable prior panel; new panel uses shared builder.'},indent=2)+'\n');print(r)
if __name__=='__main__':main()
