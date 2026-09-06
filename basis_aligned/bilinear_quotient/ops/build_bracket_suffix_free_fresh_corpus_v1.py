#!/usr/bin/env python3
"""Freeze a new pending-opener construction without loading the model."""

# BQLANE: cpu
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import tiktoken


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/prior_art/bracket_suffix_free_fresh_corpus_v1_rows.json"
ENC = tiktoken.get_encoding("gpt2")
DELIMITERS = (
    {"name":"parenthesis","open":"(","close":")"},
    {"name":"square","open":"[","close":"]"},
    {"name":"quote","open":"\"","close":"\""},
)
PAIRS = tuple((a,b) for a in range(3) for b in range(3) if a != b)
PREFIXES = ("The archivist", "A cartographer", "The botanist", "One composer", "The curator", "A geologist")
WORDS = ("anchor","basil","compass","dolphin","emerald","fossil","harbor","island","lantern","marble","orchid","pencil","quartz","saddle","violet","window","yarn","zebra")


def encode(text):
    ids=ENC.encode(text)
    if ENC.decode(ids)!=text: raise ValueError("token roundtrip failed")
    return ids


def one(text):
    ids=encode(text)
    if len(ids)!=1: raise ValueError(f"not one token: {text}")
    return ids[0]


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def main():
    if OUT.exists(): raise ValueError(f"refusing overwrite {OUT}")
    rng=random.Random(202609060031)
    rows=[]
    for left_index,right_index in PAIRS:
        left,right=DELIMITERS[left_index],DELIMITERS[right_index]
        distractor=DELIMITERS[({0,1,2}-{left_index,right_index}).pop()]
        for replicate in range(6):
            prefix=PREFIXES[replicate]
            w0,w1,w2,w3,w4=rng.sample(WORDS,5)
            leads=(f"{prefix} recorded",f"After the survey, {prefix.lower()} catalogued",f"Without closing the report, {prefix.lower()} copied")
            lead=leads[replicate%3]
            common_start=f"{lead} {distractor['open']} the {w0} and the {w1} {distractor['close']}, then opened"
            common_end=f"the {w2}, the {w3}, and the {w4} for later review"
            base=f"{common_start} {left['open']} {common_end}"
            donor=f"{common_start} {right['open']} {common_end}"
            base_ids,donor_ids=encode(base),encode(donor)
            differences=[i for i,(a,b) in enumerate(zip(base_ids,donor_ids)) if a!=b]
            if len(base_ids)!=len(donor_ids) or len(differences)!=1: raise ValueError("pair is not a one-token edit")
            coordinates={"family":"completed_distractor_then_pending_type_substitution","pair":[left_index,right_index],"replicate":replicate,"prefix":prefix,"words":[w0,w1,w2,w3,w4]}
            rows.append({"row_id":digest(coordinates),"split":"PROSPECTIVE_FRESH_V1","family_id":"completed_distractor_then_pending_type_substitution","program_role":"target","base_text":base,"donor_text":donor,"base_ids":base_ids,"donor_ids":donor_ids,"base_answer":left["close"],"donor_answer":right["close"],"base_answer_id":one(left["close"]),"donor_answer_id":one(right["close"]),"evaluation_directions":["base_to_donor","donor_to_base"],"construction_checks":{"roundtrip":True,"equal_token_length":True,"single_token_difference":True,"completed_distractor_type":distractor["name"]}})
    counts=Counter((row["base_answer_id"],row["donor_answer_id"]) for row in rows)
    if len(rows)!=36 or len(counts)!=6 or set(counts.values())!={6} or len({row["row_id"] for row in rows})!=36: raise ValueError("fresh authority balance failed")
    result={"schema":"bracket_suffix_free_fresh_corpus_rows_v1","status":"rows_frozen_outcomes_unopened","created_utc":"2026-09-06T00:31:00Z","family_id":"completed_distractor_then_pending_type_substitution","construction":"an identical completed distractor delimiter precedes a newly opened delimiter whose type alone changes","row_count":36,"endpoint_count":72,"ordered_pair_row_counts":{f"{a}->{b}":n for (a,b),n in sorted(counts.items())},"model_loaded":False,"model_forwards":0,"outcomes_opened":[],"rows":rows}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({"path":str(OUT.relative_to(ROOT)),"sha256":hashlib.sha256(OUT.read_bytes()).hexdigest(),"rows":36,"endpoints":72,"outcomes_opened":[]},sort_keys=True))


if __name__=="__main__": main()
