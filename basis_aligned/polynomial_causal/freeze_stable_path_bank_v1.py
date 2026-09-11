"""Freeze jointly stable edges before native behavioral validation; no fitting."""
import json,hashlib
from pathlib import Path
import torch


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=Path(__file__).parent;out=p/'STABLE_PATH_BANK_V1.json';ap=p/'STABLE_PATH_BANK_V1.pt';assert not out.exists() and not ap.exists()
    old=json.loads((p/'SPARSE_PATH_STABILITY_ATLAS_V1.json').read_text());center=json.loads((p/'SPARSE_PATH_CENTERED_STABILITY_V1.json').read_text())
    source=p/'COUPLED_SPARSE_PATH_CONTINUE_V1_PROGRAMS.pt';assert digest(source)==old['source_program_sha256'];assert digest(p/'SPARSE_PATH_STABILITY_ATLAS_V1.json')==center['source_atlas_sha256']
    a=next(c for c in old['comparisons'] if c['first']['mode']=='joint' and c['second']['mode']=='joint')
    b=next(c for c in center['comparisons'] if c['first']==a['first'] and c['second']==a['second'])
    selected=[k for k,(x,y) in enumerate(zip(a['matched_cosines'],b['inherited_cosines'])) if min(x,y)>=.9];pairs=[a['matching'][k] for k in selected]
    original=torch.load(source,weights_only=True,map_location='cpu')['programs'];programs=[]
    for side,key in enumerate(('first','second')):
        program=next(c for c in original if c['seed']==a[key]['seed'] and c['mode']==a[key]['mode']);ids=torch.tensor([ij[side] for ij in pairs])
        new=dict(program,original_edge_ids=ids,support=program['support'][ids],physical_writer=program['physical_writer'][:,ids]);programs.append(new)
    torch.save(dict(programs=programs,selection='Original one-to-one pairs cosine>=.9 both full and centered full-U coefficient metrics; before text validation'),ap)
    result=dict(selected_pairs=pairs,edges_per_program=len(pairs),seed_order=[q['seed'] for q in programs],
        minimum_full_cosine=min(a['matched_cosines'][k] for k in selected),minimum_centered_cosine=min(b['inherited_cosines'][k] for k in selected),
        stored_floats_per_program=[q['bank'].numel()+q['physical_writer'].numel() for q in programs],
        source_program_sha256=digest(source),artifact_sha256=digest(ap),prior_alias_receipt='SPARSE_PATH_PRIOR_SQUARE_ALIAS_V1.json',
        scope='Frozen weight-discovered candidate subprogram with native source/normalization/background dependencies. Known square aliases are retained as known components. No behavioral or circuit promotion.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
