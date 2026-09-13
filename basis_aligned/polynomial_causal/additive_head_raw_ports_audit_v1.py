"""Per-prefix sign and absolute-error audit of three-state native reconstruction."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent


def main():
    a=torch.load(P/'ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_ARTIFACT.pt',weights_only=True)
    cells=[]
    for group in range(5):
        z=a['readouts'][24*group:24*(group+1),:,0].double()
        for label,candidate,reference in [('full',4,1),('compact',5,3),('compact_vs_native_head',5,1)]:
            c=z[:,candidate]-z[:,0];r=z[:,reference]-z[:,0]
            opposite=c*r<0;zero=r==0
            cells.append(dict(group=group,comparison=label,relative_error=float((c-r).norm()/r.norm()),
                              same_nonzero_sign=int((c*r>0).sum()),opposite_nonzero_sign=int(opposite.sum()),
                              reference_zeros=int(zero.sum()),maxabs_prediction_on_reference_zero=float(c[zero].abs().max()) if zero.any() else None,
                              maxabs_error=float((c-r).abs().max())))
    result=dict(cells=cells,scope='Reused regional rows, direct conditional final-state write effects only. No new OOD or fully autonomous computation. Tiny zero cases retained.')
    (P/'ADDITIVE_HEAD_RAW_PORTS_NATIVE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
