#!/usr/bin/env python3
from collections import Counter
import json
from pathlib import Path
import tiktoken
import build_bracket_cascade_pending_ood_v1_rows as builder
def test_seventh_construction_is_balanced_aligned_and_disjoint():
 rows=builder.build_rows(tiktoken.get_encoding("gpt2").encode);root=Path(__file__).parent;old=[]
 for name in ("BRACKET_NESTED_PENDING_OOD_V1_ROWS.json","BRACKET_TRIPLE_PENDING_OOD_V1_ROWS.json","BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json","BRACKET_EMBEDDED_PENDING_OOD_V1_ROWS.json"):old.extend(json.loads((root/name).read_text())["rows"])
 assert len(rows)==72 and len({r["row_id"] for r in rows})==72 and Counter(r["program_role"] for r in rows)=={"target":36,"control":36} and not {r["base_text"] for r in rows}&{r["base_text"] for r in old};pairs=Counter()
 for r in rows:
  assert len(r["base_ids"])==len(r["donor_ids"]) and r["base_open_position"]==r["donor_open_position"]
  if r["program_role"]=="target":pairs[(r["base_answer_id"],r["donor_answer_id"])]+=1
  else:assert r["base_answer_id"]==r["donor_answer_id"]
 assert len(pairs)==3 and set(pairs.values())=={12}
def test_builder_is_model_blind():
 source=Path(builder.__file__).read_text();assert "load_bilin18" not in source and "torch" not in source and builder.OUT.name=="BRACKET_CASCADE_PENDING_OOD_V1_ROWS.json"
