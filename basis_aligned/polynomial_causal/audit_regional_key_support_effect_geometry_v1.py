"""Diagnostic scalar bounds on frozen subset writes; no gain is adopted."""
import hashlib,json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent


def geometry(v,r):
    v=v.double().flatten();r=r.double().flatten()
    assert v.norm()>0 and r.norm()>0
    cosine=(v@r)/(v.norm()*r.norm());gain=(v@r)/(v@v)
    floor=(gain*v-r).norm()/r.norm()
    assert abs(float(floor-(1-cosine.square()).clamp_min(0).sqrt()))<1e-10
    return dict(cosine=float(cosine),norm_ratio=float(v.norm()/r.norm()),diagnostic_scalar=float(gain),scalar_error_floor=float(floor),raw_error=float((v-r).norm()/r.norm()))


def main():
    path=P/'REGIONAL_KEY_SUPPORT_EFFECT_OOD_V1_ARTIFACT.pt'
    w=torch.load(path,map_location='cpu',weights_only=True)['write_vertices']
    rows=json.loads((P/'REGIONAL_SOURCE_BLOCK_OOD_V1_ROWS.json').read_text())['rows']
    cells=[]
    for arm,name in [(2,'forward'),(3,'backward')]:
        for family in range(4):
            ids=[i for i,r in enumerate(rows) if r['family']==family]
            cells.append(dict(arm=name,family=family,**geometry(w[ids,arm]-w[ids,0],w[ids,1]-w[ids,0]),
                              prefixes=[dict(row_id=i,text=rows[i]['text'],**geometry(w[i,arm]-w[i,0],w[i,1]-w[i,0])) for i in ids]))
    result=dict(cells=cells,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),artifact_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Post-result geometric diagnosis, signed scalar optimum only. No scalar is installed or scored as a repaired circuit; original 10% write/effect predictions remain failed. No claim about normalization as the cause.')
    (P/'REGIONAL_KEY_SUPPORT_EFFECT_GEOMETRY_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k!='prefixes'} for c in cells],indent=2))


if __name__=='__main__':main()
