"""Oracle finite-response lower bound for a fixed six-atom dictionary."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    prev=json.loads((P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').read_text());F=np.array(prev['plane']);atoms=[np.outer(F[:,0],F[:,0]),(np.outer(F[:,0],F[:,1])+np.outer(F[:,1],F[:,0]))/2**.5,np.outer(F[:,1],F[:,1])]
    for i,j in prev['supports']['shared2_plus3']:
        z=np.zeros((5,5));z[i,j]=z[j,i]=1/(1 if i==j else 2**.5);atoms.append(z)
    atoms=np.array(atoms)
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text());old=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    look={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    oldlook={(r['panel'],r['role'],r['family'],r['arm']):r for r in old['records'] if r['mode']=='full'}
    cases=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases'];bounds=[];normal=[]
    for c in cases:
        G=c['gradient'].numpy()[:,0]
        arms=[(np.broadcast_to(np.isin(np.arange(5),s).astype(float),G.shape),s,None) for s in native['plan']['source_sets']]
        arms += [(a.numpy(),None,name) for name,a in c['amplitudes'].items()]
        for family in dict.fromkeys(c['families']):
            ids=[i for i,f in enumerate(c['families']) if f==family];ys=[];designs=[];den=[]
            for a,s,name in arms:
                r=look[(c['panel'],c['role'],family,tuple(s))] if s is not None else oldlook[(c['panel'],c['role'],family,name)]
                target=np.array(r['target'])[:,0];ys.append(target+np.einsum('bp,bp->b',G[ids],a[ids]));den.append(max(np.linalg.norm(target),1e-30));designs.append(-.5*np.einsum('bi,kij,bj->bk',a[ids],atoms,a[ids]))
            Y=np.array(ys)/np.array(den)[:,None];D=np.array(designs)/np.array(den)[:,None,None];res=[]
            for b in range(len(ids)):
                coeff=np.linalg.lstsq(D[:,b],Y[:,b],rcond=None)[0];e=D[:,b]@coeff-Y[:,b];res.append(e);normal.append(float(abs(D[:,b].T@e).max()))
            lower=float(np.linalg.norm(res)/len(arms)**.5)
            bounds.append(dict(panel=c['panel'],role=c['role'],family=family,worst_arm_error_lower_bound=lower,excludes_ten_percent=lower>.1))
    out=dict(bounds=bounds,max_lower_bound=max(r['worst_arm_error_lower_bound'] for r in bounds),excluded_cells=sum(r['excludes_ten_percent'] for r in bounds),least_squares_normal_residual=max(normal),scope='Oracle lower bound only. Arbitrary six coefficients/context, fixed native analytic linear term, all18 finite arms. Minimum mean squared normalized arm error <= minimum maximum squared arm error. Outcomes used; not an extracted predictor. No claim against other dictionaries or different linear terms.')
    (P/'SHARED_DICTIONARY_LOWER_BOUND_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='bounds'},indent=2))
if __name__=='__main__':main()
