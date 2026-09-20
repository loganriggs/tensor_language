"""Audit semantic control sites without consulting intervention outcomes."""
import json,hashlib
from pathlib import Path
import tiktoken

def main():
    parent=Path(__file__).parent;source=parent/'SUBJECT_MODAL_PATH_FRESH_V691_ROWS.json'
    rows=json.loads(source.read_text());enc=tiktoken.get_encoding('gpt2')
    pairs=[('guard','guards'),('judge','judges'),('clerk','clerks'),('coach','coaches'),('chef','chefs'),('cook','cooks')]
    token_number={enc.encode(' '+word)[0]:n for pair in pairs for n,word in enumerate(pair)}
    assert all(len(enc.encode(' '+w))==1 for pair in pairs for w in pair)
    controls=[]
    for row in rows:
        ids=row['token_ids'];subject=row['subject_position']
        candidates=[i for i,token in enumerate(ids) if token in token_number and i!=subject]
        assert len(candidates)==1
        attractor=candidates[0]
        assert ids[subject] in token_number and token_number[ids[subject]]!=token_number[ids[attractor]]
        assert subject<attractor<row['readout_position']
        expected=389 if token_number[ids[subject]] else 318
        assert row['answer_ids'][0]==expected
        controls.append(dict(row,control_position=attractor,control_role='human noun inside relative clause; matrix subject unchanged',
                             control_token=enc.decode([ids[attractor]]).strip()))
    target=parent/'SUBJECT_ATTRACTOR_CONTROL_V696_ROWS.json';target.write_text(json.dumps(controls,indent=2)+'\n')
    result=dict(rows=len(rows),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),control_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                token_positions_verified=True,opposite_noun_numbers_verified=True,subject_answer_labels_unchanged=True,
                intervention_status='not executed; no native capability/effect claim for control edits')
    (parent/'SUBJECT_ATTRACTOR_CONTROL_V696_CPU_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':main()
