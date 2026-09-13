"""Keep full-head/text and designated-output comparisons distinct."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    regional=json.loads((P/'SPARSE_INTERACTION_FULLHEAD_REGIONAL_V1_RESULT.json').read_text())
    fineweb=json.loads((P/'SPARSE_INTERACTION_FINEWEB_V1_RESULT.json').read_text())
    rows=json.loads((P/'FIRST_TOKEN_PATH_FRESH_V1_ROWS.json').read_text())['rows']
    pairs=sorted(set((r['uk_id'],r['us_id']) for r in rows));assert len(pairs)==6
    lookup={pair:i for i,pair in enumerate(pairs)}
    a=np.array(regional['native_own_effects']);b=np.array(regional['predicted_own_effects'])
    indices=np.array([lookup[(r['uk_id'],r['us_id'])] for r in rows]);i=np.arange(96)
    aa=a[i,indices];bb=b[i,indices];designated=[]
    for group in range(4):
        ref=aa[24*group:24*(group+1)];pred=bb[24*group:24*(group+1)]
        designated.append(dict(group=group,relative_error=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),
            material_sign_reversals=int(((ref*pred<0)&(np.abs(ref)>=1e-5)).sum())))
    def aggregate(r):
        ref=np.array(r['native_own_effects']);pred=np.array(r['predicted_own_effects'])
        return dict(relative_error=float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),
                    material_sign_reversals=int(((pred*ref<0)&(np.abs(ref)>=1e-5)).sum()))
    result=dict(fullhead_regional=aggregate(regional),fullhead_fineweb=aggregate(fineweb),
        designated_regional=designated,
        scope='Full-head object and six output-pair scope matched across historical text panels. Designated regional pair slice additionally removes expanded output-probe scope. Only first24regionalprefixes overlap previous retained-port group0; baseline/port remain different. Descriptive slices preserve registered all-pair verdicts.')
    (P/'SPARSE_FULLHEAD_SCOPE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
