#!/usr/bin/env python3
"""Frozen four-behavior collateral panel for all ten Task14 program writes."""
from __future__ import annotations
from collections import Counter
import hashlib,json
from pathlib import Path

import circuit_fast_screen_candidate_polarity_state as polarity
import circuit_fast_screen_candidate_narrative_tense as tense
import circuit_fast_screen_candidate_preposition_selection as preposition
import circuit_fast_screen_candidate_voice_frame as voice

ROOT=Path(__file__).resolve().parent.parent
SCHEMA="task14_program_broad_collateral_authority_v1"
GROUPS=8
SOURCES={
 "polarity":(polarity,ROOT/"circuits/fast_screens/polarity_state_negative_vs_positive_v1_result.json","74db622b28eee6cc769f510d84c2c156a576d7a4844bffd4c3fbe67d4537eb02"),
 "narrative_tense":(tense,ROOT/"circuits/fast_screens/narrative_tense_past_vs_present_v2_result.json","5466980e1aa0a59538e4e8fcfb29457814c01e91cbe39bf41a2d42140fc7e71a"),
 "preposition_selection":(preposition,ROOT/"circuits/fast_screens/preposition_selection_on_vs_of_v1_result.json","3e57b618522f8cd1a11f5551d0691ba3921da306d8e194c041e4b219894823d8"),
 "voice_frame":(voice,ROOT/"circuits/fast_screens/voice_frame_passive_vs_active_v1_result.json","05f5d5a43993d586e499f3fc4270d8ebadb60d0a1ac9869a584b67bdc9891d58"),
}
EXPECTED_AUTHORITY_SHA256="f547177fd68314b7ebfec1ceb34ef205c44066e7a30bbd426a73defd76472c99"
class AuthorityError(ValueError):pass
def canonical_sha256(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _build_unvalidated():
 output=[]
 for behavior,(module,result_path,expected_sha) in SOURCES.items():
  if _sha(result_path)!=expected_sha:raise AuthorityError(f"{behavior} result changed")
  result=json.loads(result_path.read_text())
  if result.get("terminal")!="screen":raise AuthorityError(f"{behavior} source is not a screen")
  native={(x["row_id"],x["side"]):x for x in result["run"]["native_logits"]}
  source_rows=module.build_rows(groups=GROUPS)
  for row in source_rows:
   item=native.get((row["row_id"],"base"))
   if item is None or float(item["answer_logit"])-float(item["foil_logit"])<=0:raise AuthorityError(f"{behavior} base row lacks native capability")
   output.append({"schema":SCHEMA,"behavior":behavior,"source_task_id":row["task_id"],"source_row_id":row["row_id"],"row_id":canonical_sha256([SCHEMA,behavior,row["row_id"]]),"transform_id":row["transform_id"],"direction_id":row["direction_id"],"text":row["base_text"],"ids":row["base_ids"],"answer_id":row["base_answer_id"],"foil_id":row["base_foil_id"],"semantic_position":row["base_semantic_position"]})
 return output
def validate_rows(rows,verify_hash=True):
 rows=list(rows)
 if rows!=_build_unvalidated() or len(rows)!=128 or len({x["row_id"] for x in rows})!=128:raise AuthorityError("panel changed")
 if Counter(x["behavior"] for x in rows)!=Counter({key:32 for key in SOURCES}):raise AuthorityError("behavior balance changed")
 if any(x["semantic_position"]!=len(x["ids"])-1 or x["answer_id"]==x["foil_id"] for x in rows):raise AuthorityError("endpoint contract failed")
 digest=canonical_sha256(rows)
 if verify_hash and digest!=EXPECTED_AUTHORITY_SHA256:raise AuthorityError(f"hash changed {digest}")
 return digest
def build_rows():
 rows=_build_unvalidated();validate_rows(rows);return rows
def compile_plan():
 rows=build_rows();return {"schema":SCHEMA,"row_count":len(rows),"behaviors":list(SOURCES),"rows_per_behavior":32,"groups_per_source":GROUPS,"authority_sha256":canonical_sha256(rows),"source_result_sha256":{key:value[2] for key,value in SOURCES.items()}}
if __name__=="__main__":print(json.dumps(compile_plan(),indent=2,sort_keys=True))
