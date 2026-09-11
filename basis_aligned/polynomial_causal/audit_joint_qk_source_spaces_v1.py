"""Compare frozen native QK source spaces in common raw source coordinates."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib
from pathlib import Path
import torch


def main():
    p=Path(__file__).resolve().parent;torch.set_num_threads(2)
    r=json.loads((p/'JOINT_QK_SOURCE_BOUND_V1_RESULT.json').read_text());cache=r['cache']
    assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    s=torch.load(cache['path'],weights_only=True,map_location='cpu')
    def compare(a,b):
        c=torch.linalg.svdvals(a.T@b)
        return dict(mean_squared_principal_cosine=float(c.square().mean()),maximum_principal_cosine=float(c.max()),minimum_principal_cosine=float(c.min()))
    positions=[dict(head=h,**compare(s[f'{h}:7'],s[f'{h}:0'])) for h in range(9)]
    pairs=[dict(source_position=pos,heads=[h,k],**compare(s[f'{h}:{pos}'],s[f'{k}:{pos}'])) for pos in [7,0] for h in range(9) for k in range(h+1,9)]
    out=dict(position_comparisons=positions,head_comparisons=pairs,cache_sha256=cache['sha256'],mean_inside_fraction=sum(x['inside_fraction'] for x in r['rows'])/18,mean_mixed_fraction=sum(x['mixed_fraction'] for x in r['rows'])/18,body_forwards=0,corpus_access=False,scope='Descriptive common raw-source geometry of frozen spectral spaces. Two fixed source distances are not an OOD text test. No fit or threshold-selected circuit.')
    with (p/'JOINT_QK_SOURCE_SPACE_COMPARISON_V1_AUDIT.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(positions,indent=2))


if __name__=='__main__':main()
