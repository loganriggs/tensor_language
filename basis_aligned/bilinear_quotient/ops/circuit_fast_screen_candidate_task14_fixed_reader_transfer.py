#!/usr/bin/env python3
"""Frozen new-text authority for fixed direction-reader transfer."""
from __future__ import annotations
from collections import Counter
import hashlib,json

import circuit_battery_task14 as old_task14
import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as prior

SCHEMA="task14_fixed_reader_transfer_authority_v1"
TASK_ID="subject_verb.number_agreement"
CAPABILITY_ID="subject_verb.number_agreement.fixed_reader_transfer_capability_v1"
CAUSAL_CANDIDATE_ID="subject_verb.number_agreement.fixed_direction_reader_cross_corpus_transfer_v1"
SUBJECT_POSITION=8
ROLES=("recipient","opposite_same_lemma","same_number_different_lemma")
TEMPLATES=(("inside_above","Inside the {a1} above the {a2}, the {subject}"),
           ("above_inside","Above the {a1} inside the {a2}, the {subject}"))
NOUN_PAIRS=(("tenant","tenants"),("patron","patrons"),("master","masters"),("scout","scouts"),
 ("rebel","rebels"),("pirate","pirates"),("bishop","bishops"),("senator","senators"),
 ("deputy","deputies"),("servant","servants"),("expert","experts"),("veteran","veterans"),
 ("citizen","citizens"),("resident","residents"),("visitor","visitors"),("tourist","tourists"))
EXPECTED_AUTHORITY_SHA256="68ccb475b41e8f566dc4ac11d293429065e81b156ee19e3a4f6f229b1defb75e"

def canonical_sha256(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def _render(template,a1,a2,subject):
 text=template.format(a1=a1,a2=a2,subject=subject); ids=old_task14.ENCODING.encode(text)
 if len(ids)!=9: raise ValueError(f"not nine tokens: {text!r} {ids}")
 return text,ids
def _build_unvalidated():
 rows=[]; half=len(NOUN_PAIRS)//2
 for group in range(len(NOUN_PAIRS)):
  rn=0 if group<half else 1; within=group%half; direction="singular_to_plural" if rn==0 else "plural_to_singular"
  lexical=(0 if rn==0 else half)+(within+3)%half; a1g=(group+2)%len(NOUN_PAIRS); a2g=(group+6)%len(NOUN_PAIRS)
  state=((within%4)//2,within%2); a1=NOUN_PAIRS[a1g][state[0]]; a2=NOUN_PAIRS[a2g][state[1]]
  for tid,template in TEMPLATES:
   subjects={"recipient":NOUN_PAIRS[group][rn],"opposite_same_lemma":NOUN_PAIRS[group][1-rn],
    "same_number_different_lemma":NOUN_PAIRS[lexical][rn]}; numbers={"recipient":rn,"opposite_same_lemma":1-rn,"same_number_different_lemma":rn}; endpoints={}
   for role in ROLES:
    text,ids=_render(template,a1,a2,subjects[role]); answer=318 if numbers[role]==0 else 389
    endpoints[role]={"text":text,"ids":ids,"subject":subjects[role],"subject_number":"singular" if numbers[role]==0 else "plural",
     "answer_id":answer,"foil_id":389 if answer==318 else 318}
   identity=[SCHEMA,group,tid,state,endpoints]; rows.append({"schema":SCHEMA,"task_id":TASK_ID,
    "capability_id":CAPABILITY_ID,"causal_candidate_id":CAUSAL_CANDIDATE_ID,"phase":"PROSPECTIVE",
    "group_number":group,"row_id":canonical_sha256(identity),"template_id":tid,"direction_id":direction,
    "attractor_state":list(state),"subject_position":SUBJECT_POSITION,"endpoints":endpoints})
 return rows

def validate_rows(rows,verify_hash=True):
 rows=list(rows)
 if rows!=_build_unvalidated() or len(rows)!=32 or len({x["row_id"] for x in rows})!=32: raise ValueError("authority changed")
 if sorted(Counter((x["direction_id"],x["template_id"]) for x in rows).values())!=[8]*4: raise ValueError("balance changed")
 prior_vocab,prior_prompts,prior_tokens=prior._prior_material()
 prior_vocab.update(w for pair in prior.NOUN_PAIRS for w in pair)
 for row in prior.build_rows():
  for e in row["endpoints"].values(): prior_prompts.add(e["text"]); prior_tokens.add(tuple(e["ids"]))
 forms={w for pair in NOUN_PAIRS for w in pair}
 if len(forms)!=32 or forms&prior_vocab or any(len(old_task14.ENCODING.encode(" "+w))!=1 for w in forms): raise ValueError("noun novelty/tokenization failed")
 prompts=[]; tokens=[]
 for row in rows:
  r,o,l=(row["endpoints"][x] for x in ROLES)
  if r["subject_number"]==o["subject_number"] or r["subject_number"]!=l["subject_number"] or r["subject"]==l["subject"]: raise ValueError("endpoint relation failed")
  for alt in (o,l):
   if [i for i,p in enumerate(zip(r["ids"],alt["ids"])) if p[0]!=p[1]]!=[SUBJECT_POSITION]: raise ValueError("endpoint delta failed")
  for e in row["endpoints"].values():
   answer=" is" if e["answer_id"]==318 else " are"
   if old_task14.ENCODING.encode(e["text"]+answer)!=e["ids"]+[e["answer_id"]]: raise ValueError("continuation failed")
   prompts.append(e["text"]); tokens.append(tuple(e["ids"]))
 if len(set(prompts))!=96 or len(set(tokens))!=96 or set(prompts)&prior_prompts or set(tokens)&prior_tokens: raise ValueError("prompt novelty failed")
 digest=canonical_sha256(rows)
 if verify_hash and digest!=EXPECTED_AUTHORITY_SHA256: raise ValueError(f"hash changed {digest}")
 return digest
def build_rows():
 rows=_build_unvalidated(); validate_rows(rows); return rows
def compile_plan():
 rows=build_rows(); return {"schema":SCHEMA,"task_id":TASK_ID,"capability_id":CAPABILITY_ID,
  "causal_candidate_id":CAUSAL_CANDIDATE_ID,"row_count":len(rows),"prompt_endpoints":96,
  "templates":[x[0] for x in TEMPLATES],"authority_sha256":canonical_sha256(rows)}
if __name__=="__main__": print(json.dumps(compile_plan(),indent=2,sort_keys=True))
