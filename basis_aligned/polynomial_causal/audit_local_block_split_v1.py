"""Reusable saved-outcome sign/absolute audit for the eight-arm local block split."""
from pathlib import Path
import json
import sys
import torch
P=Path(__file__).resolve().parent


def main(stem):
    a=torch.load(P/(stem+'_ARTIFACT.pt'),weights_only=True)['measures'].double();cells=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
        z=a[lo:hi,:,0];full=z[:,4]-z[:,5];mlp=z[:,6]-z[:,5];att=z[:,7]-z[:,5]
        cells.append(dict(cell=label,mlp_same_sign=int((mlp*full>0).sum()),mlp_opposite_sign=int((mlp*full<0).sum()),
            full_zero_count=int((full==0).sum()),meanabs_full_effect=float(full.abs().mean()),
            maxabs_mlp_prediction_error=float((mlp-full).abs().max()),maxabs_effect_composition_error=float((full-mlp-att).abs().max()),
            attention_effect_norm_over_full=float(att.norm()/full.norm())))
    result=dict(cells=cells,scope='Saved native outcome audit, no new fitting. Sign and absolute errors complement registered group-relative bars; no universal circuit/task claim.')
    (P/(stem+'_AUDIT.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main(sys.argv[1])
