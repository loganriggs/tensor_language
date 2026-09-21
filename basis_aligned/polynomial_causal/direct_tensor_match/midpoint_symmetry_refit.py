"""Conditional output-refit screen; no held native panel is used for fitting."""
from pathlib import Path
import json
import torch

def main():
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    p = Path(__file__).resolve().parent
    out = p / 'MIDPOINT_SYMMETRY_REFIT_V1.json'
    assert not out.exists()
    rows = torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt', weights_only=True)
    n, m, y = [rows[k].flatten(0,1).double() for k in ('n','m','y')]
    S = torch.load(p/'MIDPOINT_CENTERED_V1.pt', weights_only=True)['metric_sqrt'].double()
    e = {k:v.double() for k,v in torch.load(p/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt', weights_only=True)['products512'].items()}
    A, B = e['Pn']@e['Tn'], e['Pm']@e['Tm']
    C = e['output_basis']@e['output_core']
    forward, reverse = (n@A)*(m@B), (m@A)*(n@B)
    cut = 24*rows['n'].shape[1]
    ym = y[:cut].mean(0)
    target = y-ym
    records = []
    for name, phi in [('ordered', forward), ('symmetric', (forward+reverse)/2)]:
        phi = phi-phi[:cut].mean(0)
        scale = phi[:cut].square().mean(0).sqrt().clamp_min(1e-12)
        X = phi/scale
        prior = scale[:,None]*C.T
        gram = X[:cut].T@X[:cut]/cut
        cross = X[:cut].T@(target[:cut]-X[:cut]@prior)/cut
        ev, V = torch.linalg.eigh(gram)
        projected = V.T@cross
        def errors(W):
            return {label:float(((X[sl]@W-target[sl])@S).norm()/(target[sl]@S).norm()) for label,sl in [('train',slice(None,cut)),('validation',slice(cut,None))]}
        records.append(dict(family=name,ridge=None,**errors(prior)))
        for ridge in [.001,.01,.1,1.,10.,100.]:
            W = prior+V@(projected/(ev.clamp_min(0)+ridge)[:,None])
            records.append(dict(family=name,ridge=ridge,**errors(W)))
    # A symmetric fit can recover the role-support toy even when averaging fails.
    K = torch.tensor([[0.,1.],[1.,0.]],dtype=torch.float64)
    approx = torch.tensor([[0.,1.],[0.,0.]],dtype=torch.float64)
    g = torch.Generator().manual_seed(9153)
    u,v = torch.randn(2,100,generator=g,dtype=torch.float64)
    xn,xm = torch.stack((u,0*u),1),torch.stack((0*v,v),1)
    truth = torch.einsum('ni,ij,nj->n',xn,K,xm)
    sym = (approx+approx.T)/2
    pred = torch.einsum('ni,ij,nj->n',xn,sym,xm)
    coefficient = pred@truth/(pred@pred)
    toy = dict(refitted_scale=float(coefficient),relative_error=float((coefficient*pred-truth).norm()/truth.norm()))
    assert toy['relative_error']<1e-12
    result = dict(records=records,toy=toy,scope='Conditional calibration split: first 24 documents fit, last 8 validation; features previously learned using all 32, so not independent generalization. Original native reduced-output target, fixed vocabulary metric. Symmetric family uses two products per retained pair (1024 vs ordered512). All means and scales fitted on first24. Ridge anchors to original pruned output map; only output map refitted. Not full symmetry-constrained factor learning.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__ == '__main__':
    main()
