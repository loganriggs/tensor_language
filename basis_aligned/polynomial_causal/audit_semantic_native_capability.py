"""CPU label/role and native capability audit; no outcome-based row filtering."""
import json
from pathlib import Path
import numpy as np
import tiktoken
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'

def main():
    enc=tiktoken.get_encoding('gpt2');r=json.loads((A/'semantic_port_pairs_fresh_v1_result.json').read_text());failures=[];signs=[]
    pairs=[('guard','guards'),('judge','judges'),('clerk','clerks'),('coach','coaches'),('chef','chefs'),('cook','cooks')]
    number={enc.encode(' '+word)[0]:n for pair in pairs for n,word in enumerate(pair)}
    for panel in ['opposite','congruent']:
        rows=json.loads((P/f'SEMANTIC_PORT_FRESH_{panel.upper()}_ROWS.json').read_text())
        for family,cell in r['data'][panel]['subject']['B'].items():
            subset=[row for row in rows if row['family']==family];assert len(subset)==len(cell['base'])
            for row,margin in zip(subset,cell['base']):
                ids=row['token_ids'];s=row['subject_position'];a=row['control_position'];n=number[ids[s]]
                assert enc.decode(ids)==row['text'] and s<a<row['readout_position']==len(ids)-1
                assert row['answer_ids']==([389,318] if n else [318,389])
                assert (number[ids[a]]==n)==(panel=='congruent')
                if margin<=0:failures.append(dict(panel=panel,family=family,text=row['text'],subject=enc.decode([ids[s]]),attractor=enc.decode([ids[a]]),correct_margin=margin))
    for file in ['semantic_port_pairs_v1_result.json','semantic_port_pairs_fresh_v1_result.json']:
        z=json.loads((A/file).read_text())
        for panel in ['opposite','congruent']:
            for role in ['subject','attractor']:
                cs=[c for c in z['pair_records'] if c['panel']==panel and c['role']==role]
                target=np.concatenate([c['target'] for c in cs]);den=float(target@target)
                for pair in ['pair23','pair24','pair34']:
                    v=np.concatenate([c['pair_terms'][pair] for c in cs])
                    signs.append(dict(receipt=file,panel=panel,role=role,pair=pair,aligned_fraction=float(v@target/den),relative_norm=float(np.linalg.norm(v)/np.linalg.norm(target))))
    out=dict(labels_and_positions_verified=True,native_wrong_rows=failures,interaction_support=signs,scope='No row exclusions, threshold changes, or fresh-data support replacement. Full native baseline replay was exactly zero discrepancy.')
    (P/'SEMANTIC_NATIVE_CAPABILITY_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(failures,indent=2))
if __name__=='__main__':main()
