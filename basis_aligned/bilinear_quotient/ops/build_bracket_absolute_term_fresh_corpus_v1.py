#!/usr/bin/env python3
"""Freeze a fourth pending-opener construction for absolute-term validation."""

# BQLANE: cpu
from __future__ import annotations
import hashlib,json,random
from collections import Counter
from pathlib import Path
from build_bracket_suffix_free_fresh_corpus_v1 import DELIMITERS,PAIRS,digest,encode,one

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"circuits/prior_art/bracket_absolute_term_fresh_corpus_v1_rows.json"
PREFIXES=("The apothecary","A blacksmith","The choreographer","One diplomat","The electrician","A watchmaker")
WORDS=("acorn","brooch","canyon","drum","easel","flute","goblet","hinge","inkwell","jewel","kettle","lilac","magnet","napkin","obelisk","plum","rattle","thimble")

def main():
 if OUT.exists():raise ValueError(f"refusing overwrite {OUT}")
 rng=random.Random(202609060037);rows=[]
 for li,ri in PAIRS:
  left,right=DELIMITERS[li],DELIMITERS[ri];distractor=DELIMITERS[({0,1,2}-{li,ri}).pop()]
  for replicate,prefix in enumerate(PREFIXES):
   w0,w1,w2,w3,w4=rng.sample(WORDS,5)
   starts=(f"Although {prefix.lower()} had closed",f"After {prefix.lower()} finished",f"Because {prefix.lower()} reviewed")
   common=f"{starts[replicate%3]} {distractor['open']} the {w0} and the {w1} {distractor['close']}, a separate final note began"
   tail=f"the {w2}, the {w3}, and the {w4} without an ending"
   base=f"{common} {left['open']} {tail}";donor=f"{common} {right['open']} {tail}"
   bi,di=encode(base),encode(donor);diff=[i for i,(a,b) in enumerate(zip(bi,di)) if a!=b]
   if len(bi)!=len(di) or len(diff)!=1:raise ValueError("not a one-token edit")
   coord={"family":"subordinate_completed_distractor_pending_type_substitution","pair":[li,ri],"replicate":replicate,"prefix":prefix,"words":[w0,w1,w2,w3,w4]}
   rows.append({"row_id":digest(coord),"split":"PROSPECTIVE_ABSOLUTE_TERM_V1","family_id":"subordinate_completed_distractor_pending_type_substitution","program_role":"target","base_text":base,"donor_text":donor,"base_ids":bi,"donor_ids":di,"base_answer":left["close"],"donor_answer":right["close"],"base_answer_id":one(left["close"]),"donor_answer_id":one(right["close"]),"evaluation_directions":["base_to_donor","donor_to_base"],"construction_checks":{"roundtrip":True,"equal_token_length":True,"single_token_difference":True,"completed_distractor_type":distractor["name"]}})
 counts=Counter((r["base_answer_id"],r["donor_answer_id"]) for r in rows)
 if len(rows)!=36 or len(counts)!=6 or set(counts.values())!={6}:raise ValueError("balance failed")
 value={"schema":"bracket_absolute_term_fresh_corpus_rows_v1","status":"rows_frozen_outcomes_unopened","created_utc":"2026-09-06T00:37:00Z","family_id":"subordinate_completed_distractor_pending_type_substitution","construction":"a subordinate clause closes an irrelevant delimiter before a separate final note opens the active pending delimiter","row_count":36,"endpoint_count":72,"ordered_pair_row_counts":{f"{a}->{b}":n for (a,b),n in sorted(counts.items())},"model_loaded":False,"model_forwards":0,"outcomes_opened":[],"rows":rows}
 OUT.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n");print(json.dumps({"path":str(OUT.relative_to(ROOT)),"sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),"rows":36,"endpoints":72,"outcomes_opened":[]},sort_keys=True))
if __name__=="__main__":main()
