#!/usr/bin/env python3
"""Author unseen contexts for the instruction-only head-cross intervention."""
import hashlib, json
from pathlib import Path
import tiktoken
from regional_city_article_check_v1 import validate_city_articles
from regional_cue_row_check_v1 import validate

TEMPLATES=(
    'A curator in {city} catalogued a private notebook. Follow its spelling while completing this quotation: "Their recent',
    'A collector from {city} saved the author\'s untouched draft. Continue in precisely the same regional style: "Their recent',
)
CITY_PAIRS=(("Glasgow","Seattle"),("London","Dallas"))
ENDPOINTS=(("colour","color"),("centre","center"),("travelled","traveled"),("theatre","theater"),("organise","organize"),("neighbour","neighbor"))

def main():
    root=Path(__file__).resolve().parent; out=root/"SETTING2_REGIONAL_INSTRUCTION_CROSS_FRESH_V1_ROWS.json"
    if out.exists(): raise FileExistsError(out)
    enc=tiktoken.get_encoding("gpt2"); rows=[]
    for family,template in enumerate(TEMPLATES):
        for pair,(uk_city,us_city) in enumerate(CITY_PAIRS):
            for endpoint,(uk_word,us_word) in enumerate(ENDPOINTS):
                targets=[enc.encode(" "+word) for word in (uk_word,us_word)]
                if any(len(x)!=1 for x in targets): raise ValueError("endpoint must be one token")
                for cue,city in (("British",uk_city),("American",us_city)):
                    row={"family":family,"template":family,"pair":pair,"endpoint":endpoint,"cue":cue,"city":city,"text":template.format(city=city),"uk_id":targets[0][0],"us_id":targets[1][0],"control_ids":[3797,3290]}; row["ids"]=enc.encode(row["text"]); row["row_id"]=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(",",":")).encode()).hexdigest(); rows.append(row)
    if len(rows)!=48: raise ValueError("expected 48 rows")
    checks=validate(rows); validate_city_articles(rows); prior=set()
    for path in root.glob("*ROWS.json"):
        if path==out: continue
        try: prior.update(tuple(r.get("ids",())) for r in json.loads(path.read_text()).get("rows",()))
        except (json.JSONDecodeError,AttributeError,TypeError): pass
    overlap=sum(tuple(r["ids"]) in prior for r in rows)
    if overlap: raise ValueError(f"{overlap} prior contexts")
    out.write_text(json.dumps({"schema":"setting2_regional_instruction_cross_fresh_rows_v1","selection":"Templates, cities, and endpoints fixed without model execution, activations, or scores.","templates":TEMPLATES,"city_pairs":CITY_PAIRS,"endpoints":ENDPOINTS,"prior_context_overlap":overlap,"row_checks":checks,"rows":rows},indent=2)+"\n")
    print(json.dumps({"rows":len(rows),"pairs":len(rows)//2,"lengths":sorted({len(r['ids']) for r in rows}),"prior_context_overlap":overlap}))

if __name__=="__main__": main()
