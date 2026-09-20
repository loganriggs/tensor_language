"""Finite interaction fidelity and early-block source-cross path, CPU only."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'five_source_full_span_v1_result.json').read_text());lookup={(c['panel'],c['role'],c['family'],tuple(c['source_set'])):c for c in r['records'] if c['mode']=='full'}
    interactions=[]
    for key,c in lookup.items():
        if len(key[-1])!=2:continue
        i,j=key[-1];ci=lookup[key[:-1]+((i,),)];cj=lookup[key[:-1]+((j,),)]
        y=np.array(c['target'])-ci['target']-np.array(cj['target']);pred=np.array(c['prediction'])-ci['prediction']-np.array(cj['prediction']);budget=c['number_norm'];norms=np.linalg.norm(y,axis=0);errors=np.linalg.norm(pred-y,axis=0)
        interactions.append(dict(panel=c['panel'],role=c['role'],family=c['family'],source_set=[i,j],native_interaction_over_pair_number=(norms/budget).tolist(),error_over_pair_number=(errors/budget).tolist(),own_interaction_errors=(errors/np.maximum(norms,1e-30)).tolist(),material=(norms/budget>=.01).tolist()))
    package=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True);paths=[]
    for case in package['cases']:
        H=sum(v.numpy() for v in case['terms'].values());early=case['terms']['mlp11'].numpy()+case['terms']['attention11'].numpy();modified=H.copy();modified[:,:,:2,2:]=early[:,:,:2,2:];modified[:,:,2:,:2]=early[:,:,2:,:2];G=case['gradient'].numpy()
        for selected in r['plan']['source_sets']:
            amp=np.zeros(5);amp[selected]=1;pred=-np.einsum('bop,p->bo',G,amp)-.5*np.einsum('p,bopq,q->bo',amp,modified,amp)
            for family in dict.fromkeys(case['families']):
                ids=[i for i,f in enumerate(case['families']) if f==family];c=lookup[(case['panel'],case['role'],family,tuple(selected))];y=np.array(c['target']);errors=np.linalg.norm(pred[ids]-y,axis=0)/c['number_norm']
                paths.append(dict(panel=case['panel'],role=case['role'],family=family,source_set=selected,number_error=float(errors[0]),modal_errors=errors[1:].tolist()))
    material=[c for c in interactions if c['material'][0]]
    result=dict(material_number_interaction_cells=len(material),max_material_number_interaction_relative_error=max(c['own_interaction_errors'][0] for c in material),max_number_interaction_error_over_pair_number=max(c['error_over_pair_number'][0] for c in interactions),early_AB_only=dict(max_number_error=max(c['number_error'] for c in paths),max_modal_error=max(max(c['modal_errors']) for c in paths),passes=all(c['number_error']<=.1 and max(c['modal_errors'])<=.05 for c in paths)),interactions=interactions,path_records=paths,scope='Opened finite interaction and fixed early-A/B curvature path audit. Other source self/cross curvature retained; all native Jacobians/readers retained. Not a primal module ablation.')
    (P/'SOURCE_CROSS_CURVATURE_CPU_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ['interactions','path_records']})
if __name__=='__main__':main()
