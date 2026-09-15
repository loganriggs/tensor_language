#!/usr/bin/env python3
"""Freeze a sixth embedded-pending bracket construction without model access."""
from __future__ import annotations
from collections import Counter
import hashlib,json
from pathlib import Path
import tiktoken

HERE=Path(__file__).resolve().parent; OUT=HERE/"BRACKET_EMBEDDED_PENDING_OOD_V1_ROWS.json"
MARKS={"parenthesis":{"open":"(","close":")","open_id":357},"square":{"open":"[","close":"]","open_id":685},"quote":{"open":'"',"close":'"',"open_id":366}}
UNORDERED=(("parenthesis","square"),("parenthesis","quote"),("square","quote"))
GROUPS=(("astronomy syllabus","telescope","galaxy","observation"),("culinary handbook","sauce","spice","preparation"),("mining summary","shaft","ore","survey"),("dance notation","gesture","rhythm","sequence"),("insurance memo","policy","claim","review"),("botanical index","fern","meadow","specimen"),("navigation lesson","compass","bearing","landfall"),("architecture note","facade","column","renovation"),("textile catalogue","fiber","pattern","weaving"),("public health brief","clinic","screening","outcome"),("energy forecast","turbine","storage","capacity"),("history seminar","archive","treaty","interpretation"))
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def build_rows(token):
 answers={name:token(spec["close"])[0] for name,spec in MARKS.items()}; rows=[]
 for gi,(document,a,b,c) in enumerate(GROUPS):
  for left,right in UNORDERED:
   middle=next(name for name in MARKS if name not in (left,right))
   def text(inner): return f"In the {document}, {{ the introduction remained open before {MARKS[middle]['open']} an annotation started; within it {MARKS[inner]['open']} the {a}, the {b}, and the {c} awaited completion"
   base,donor=text(left),text(right); bi,di=token(base),token(donor); assert len(bi)==len(di)
   bo=max(i for i,v in enumerate(bi) if v==MARKS[left]["open_id"]); do=max(i for i,v in enumerate(di) if v==MARKS[right]["open_id"]); assert bo==do and [i for i,z in enumerate(zip(bi,di)) if z[0]!=z[1]]==[bo]
   row={"family_id":"embedded_pending_stack_top","program_role":"target","group_index":gi,"base_text":base,"donor_text":donor,"base_ids":bi,"donor_ids":di,"base_answer_id":answers[left],"donor_answer_id":answers[right],"base_open_position":bo,"donor_open_position":do,"base_type":left,"donor_type":right,"outer_type":"brace","middle_type":middle}; row["row_id"]=canonical(row); rows.append(row)
  for inner in MARKS:
   middles=[name for name in MARKS if name!=inner]
   def text(middle): return f"In the {document}, {{ the introduction remained open before {MARKS[middle]['open']} an annotation started; within it {MARKS[inner]['open']} the {a}, the {b}, and the {c} awaited completion"
   base,donor=text(middles[0]),text(middles[1]); bi,di=token(base),token(donor); assert len(bi)==len(di)
   bo=max(i for i,v in enumerate(bi) if v==MARKS[inner]["open_id"]); do=max(i for i,v in enumerate(di) if v==MARKS[inner]["open_id"]); assert bo==do and len([i for i,z in enumerate(zip(bi,di)) if z[0]!=z[1]])==1
   row={"family_id":"embedded_middle_change_inner_fixed","program_role":"control","group_index":gi,"base_text":base,"donor_text":donor,"base_ids":bi,"donor_ids":di,"base_answer_id":answers[inner],"donor_answer_id":answers[inner],"base_open_position":bo,"donor_open_position":do,"inner_type":inner,"outer_type":"brace","base_middle_type":middles[0],"donor_middle_type":middles[1]}; row["row_id"]=canonical(row); rows.append(row)
 return rows
def main():
 if OUT.exists(): raise FileExistsError(OUT)
 rows=build_rows(tiktoken.get_encoding("gpt2").encode); counts=Counter(r["program_role"] for r in rows); pairs=Counter()
 for r in rows:
  if r["program_role"]=="target": pairs[(r["base_answer_id"],r["donor_answer_id"])]+=1; pairs[(r["donor_answer_id"],r["base_answer_id"])]+=1
 assert len(rows)==72 and counts=={"target":36,"control":36} and len(pairs)==6 and set(pairs.values())=={12}
 payload={"schema":"bracket_embedded_pending_ood_v1_rows","model_loaded":False,"outcomes_opened":[],"row_count":72,"endpoint_count":144,"counts":dict(counts),"ordered_pair_counts":{f"{a}->{b}":n for (a,b),n in sorted(pairs.items())},"rows":rows}; payload["row_manifest_sha256"]=canonical(rows); OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print(json.dumps({k:payload[k] for k in ("row_count","endpoint_count","counts","ordered_pair_counts","row_manifest_sha256")},indent=2))
if __name__=="__main__": main()
