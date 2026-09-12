"""Descriptive red-team: collinearity and cancellation, no fit or changed bars."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(2)
    data=torch.load(P/'REGIONAL_ROUTING_CONSUMER_REUSE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows']
    w=data['write_vertices'];m=data['arm_margins'];cells=[]
    def cosine(a,b):return float((a.flatten()@b.flatten())/(a.norm()*b.norm()).clamp_min(1e-30))
    for family in range(4):
        ids=[i for i,r in enumerate(rows) if r['family']==family]
        first=w[ids,1]-w[ids,0];current=w[ids,2]-w[ids,0]
        ef=w[ids,4]-w[ids,1];ec=w[ids,5]-w[ids,2]
        pair=torch.stack([first.flatten()/first.norm(),current.flatten()/current.norm()],1)
        sv=torch.linalg.svdvals(pair)
        fm=m[1,ids,0]-m[0,ids,0];cm=m[2,ids,0]-m[0,ids,0]
        cells.append(dict(family=family,write_cosine=cosine(first,current),effect_cosine=cosine(fm,cm),
            normalized_pair_singular_ratio=float(sv[1]/sv[0]),
            approximation_error_cosine=cosine(ef,ec),
            error_sum_over_individual_norm_sum=float((ef+ec).norm()/(ef.norm()+ec.norm())),
            native_sum_over_individual_norm_sum=float((first+current).norm()/(first.norm()+current.norm()))))
    out=dict(cells=cells,scope='Descriptive saved-output geometry only. Collinearity can limit evidence for distinct consumers. Cancellation cannot rescue a failed individual branch; all individual branches already passed their frozen bars.')
    (P/'REGIONAL_CONSUMER_GEOMETRY_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':main()
