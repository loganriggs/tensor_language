#!/usr/bin/env python3
# BQLANE: cpu
"""Build fresh nonadjacent digit/word +1 sequence pairs without a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken


HERE=Path(__file__).resolve().parent
OUT=HERE/"NUMERIC_SEQUENCE_NONADJACENT_OOD_V1_ROWS.json"
ENC=tiktoken.get_encoding("gpt2")
WORDS=("anchor","birch","cobalt","drizzle","falcon","glacier","harbor","lantern")
NUMBER_WORD={2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",
             9:"nine",10:"ten",11:"eleven"}


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def endpoint(text,answer,values,representation):
    ids=ENC.encode(text); answer_ids=ENC.encode(answer)
    if ENC.decode(ids)!=text or len(answer_ids)!=1: raise ValueError((text,answer,answer_ids))
    return {"text":text,"ids":ids,"answer":answer,"answer_id":answer_ids[0],
            "values":list(values),"representation":representation}


def render(values,word,representation,headered):
    shown=values if representation=="digit" else tuple(NUMBER_WORD[x] for x in values)
    body=", ".join(str(x) for x in shown)+","
    return (f"Continue the {word} count: {body}" if headered else f"The {word} series is {body}")


def main():
    rows=[]
    for construction,headered in (("fresh_series",False),("fresh_instruction",True)):
        for representation in ("digit","number_word"):
            for index,word in enumerate(WORDS):
                low_start=2+(index%2); high_start=low_start+5
                low=tuple(range(low_start,low_start+3)); high=tuple(range(high_start,high_start+3))
                low_answer=(f" {low[-1]+1}" if representation=="digit" else " "+NUMBER_WORD[low[-1]+1])
                high_answer=(f" {high[-1]+1}" if representation=="digit" else " "+NUMBER_WORD[high[-1]+1])
                base=endpoint(render(low,word,representation,headered),low_answer,low,representation)
                donor=endpoint(render(high,WORDS[-1-index],representation,headered),high_answer,high,representation)
                row_id=digest([construction,representation,index,base["text"],donor["text"]])
                rows.append({"row_id":row_id,"construction":construction,"representation":representation,
                             "relation":"plus_one","direction_policy":"evaluate_both","base":base,"donor":donor})
    result={"schema":"numeric_sequence_nonadjacent_ood_v1_rows","row_count":len(rows),
            "endpoint_count":2*len(rows),"row_manifest_sha256":digest(rows),"rows":rows}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result[k] for k in ("schema","row_count","endpoint_count","row_manifest_sha256")},indent=2))


if __name__=="__main__":main()
