"""Change only the relative-clause noun's number, preserving subject labels."""
import json,hashlib
from pathlib import Path
import tiktoken

def main():
    p=Path(__file__).parent;rows=json.loads((p/'SUBJECT_ATTRACTOR_CONTROL_V696_ROWS.json').read_text())
    enc=tiktoken.get_encoding('gpt2');pairs=[('guard','guards'),('judge','judges'),('clerk','clerks'),('coach','coaches'),('chef','chefs'),('cook','cooks')]
    tokens=[[enc.encode(' '+w)[0] for w in pair] for pair in pairs]
    lookup={token:(pair,n) for pair in tokens for n,token in enumerate(pair)}
    for row in rows:
        ids=row['token_ids'];s=row['subject_position'];a=row['control_position']
        _,number=lookup[ids[s]];pair,anumber=lookup[ids[a]];assert number!=anumber
        previous=ids.copy();ids[a]=pair[number]
        assert sum(x!=y for x,y in zip(ids,previous))==1
        row['text']=enc.decode(ids);row['control_token']=enc.decode([ids[a]]).strip()
        row['control_number']='same as subject';row['paired_opposite_token_ids']=previous
        assert row['answer_ids'][0]==(389 if number else 318)
    out=p/'SUBJECT_CONGRUENT_ATTRACTOR_V697_ROWS.json';out.write_text(json.dumps(rows,indent=2)+'\n')
    print(len(rows),hashlib.sha256(out.read_bytes()).hexdigest())
if __name__=='__main__':main()
