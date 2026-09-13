"""Exact ordered boundary telescope for the inherited pre-block17 effect."""
from pathlib import Path
import json
import torch
P=Path(__file__).resolve().parent


def main():
    a=torch.load(P/'CROSSFIRST_BOUNDARY_CURVE_V1_ARTIFACT.pt',weights_only=True)['measures'].double()
    cells=[]
    for lo,hi,label in [(24*k,24*(k+1),'regional'+str(k)) for k in range(4)]+[(96+16*k,112+16*k,'newline'+str(k)) for k in range(4)]:
        z=a[lo:hi,:,0];inherited=z[:,2]-z[:,11]
        terms={'through_9':z[:,2]-z[:,4]}
        terms.update({str(layer):z[:,layer-6]-z[:,layer-5] for layer in range(10,17)})
        assert float((sum(terms.values())-inherited).abs().max())<1e-12
        metrics={name:dict(norm_over_inherited=float(v.norm()/inherited.norm()),aligned_over_inherited=float((v*inherited).sum()/inherited.square().sum())) for name,v in terms.items()}
        early=terms['through_9']+terms['10']+terms['11']
        late=sum(terms[str(layer)] for layer in range(12,17))
        cells.append(dict(cell=label,terms=metrics,through11_prediction_error=float((early-inherited).norm()/inherited.norm()),
                          blocks10_11_only_error=float((terms['10']+terms['11']-inherited).norm()/inherited.norm()),
                          through11_cosine=float((early*inherited).sum()/(early.norm()*inherited.norm())),
                          later12_16_meanabs_effect=float(late.abs().mean())))
    result=dict(cells=cells,scope='Exact telescope of ordered additive-state reset outcomes on reused rows. Each adjacent contrast has its own additive-input/background convention; these are not independent physical component removals or a deployed predictor.')
    (P/'INHERITED_INTERACTION_BOUNDARY_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    for c in cells:print(c)


if __name__=='__main__':main()
