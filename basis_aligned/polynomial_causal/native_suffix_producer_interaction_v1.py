"""Red-team producer interaction using the saved exact native reference.
A approximate/native cross relative RMS<=.25 each task family.
B native meanabs cross>=.1meanabs joint residual effect each task family.
Includes normalizations and final tail; not attribution to a pure quadratic term.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=Path(__file__).parent;source=p/'NATIVE_SUFFIX_MLP16_PRODUCER_V1.json';out=p/'NATIVE_SUFFIX_PRODUCER_INTERACTION_V1.json'
    assert not out.exists();prior=json.loads(source.read_text());assert prior['pred_a'];cells=[]
    for family in ('A1','A2'):
        rows=[r for r in prior['records'] if r['family']==family]
        a=np.array([r['approximate_effects'] for r in rows]);e=np.array([r['exact_effects'] for r in rows])
        ac=a[:,1]-a[:,2]-a[:,3];ec=e[:,1]-e[:,2]-e[:,3]
        cells.append(dict(family=family,n=len(rows),native_cross_meanabs=float(np.mean(np.abs(ec))),
            native_residual_effect_meanabs=float(np.mean(np.abs(e[:,1]))),native_relative_cross=float(np.mean(np.abs(ec))/np.mean(np.abs(e[:,1]))),
            approximation_relative_rms=float(np.linalg.norm(ac-ec)/np.linalg.norm(ec)),
            sign_agreement=float(np.mean(np.sign(ac)==np.sign(ec)))))
    result=dict(pred_a=all(c['approximation_relative_rms']<=.25 for c in cells),pred_b=all(c['native_relative_cross']>=.1 for c in cells),
        cells=cells,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Exact-reference interaction corroboration; original producer B/C/D misses remain. Does not isolate input normalization, bilinear numerator or output-tail causes.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
