"""Fit frozen dictionary to analytic responses before loading native outcomes."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from derivative_only_minimax import fit
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    torch.set_num_threads(2)
    prior=json.loads((P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').read_text());F=np.array(prior['plane']);support=prior['supports']['shared2_plus3']
    atoms=[np.outer(F[:,0],F[:,0]),(np.outer(F[:,0],F[:,1])+np.outer(F[:,1],F[:,0]))/2**.5,np.outer(F[:,1],F[:,1])]
    for i,j in support:
        z=np.zeros((5,5));z[i,j]=z[j,i]=1/(1 if i==j else 2**.5);atoms.append(z)
    atoms=np.array(atoms);sets=[[i] for i in range(5)]+[[i,j] for i in range(5) for j in range(i+1,5)]
    cases=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases'];frozen=[];checks=[];solver_diagnostics=[]
    for c in cases:
        G=c['gradient'].numpy();H=c['hessian_reference'].numpy()
        arms=[(str(s),np.broadcast_to(np.isin(np.arange(5),s).astype(float),(len(G),5)),s) for s in sets]
        arms += [(name,a.numpy(),None) for name,a in c['amplitudes'].items()]
        amplitudes=np.stack([a for _,a,_ in arms]);coeff=np.zeros((len(G),4,6))
        for family in dict.fromkeys(c['families']):
            ids=[i for i,f in enumerate(c['families']) if f==family]
            fitted,errors=fit(G[ids],H[ids],amplitudes[:,ids],atoms,solver_diagnostics);coeff[ids]=fitted
            checks.append(dict(panel=c['panel'],role=c['role'],family=family,analytic_errors=errors))
        R=np.einsum('bok,kij->boij',coeff,atoms)
        pred=-np.einsum('bop,abp->abo',G,amplitudes)-.5*np.einsum('abp,bopq,abq->abo',amplitudes,R,amplitudes)
        frozen.append((c,arms,coeff,pred))
    before=hashlib.sha256(b''.join(c.tobytes() for _,_,c,_ in frozen)).hexdigest()
    # Outcomes are first loaded only after every fitted coefficient is frozen.
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text());old=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    look={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    oldlook={(r['panel'],r['role'],r['family'],r['arm']):r for r in old['records'] if r['mode']=='full'}
    scores=[];decomposition=[]
    for c,arms,coeff,pred in frozen:
        for index,(name,_,selected) in enumerate(arms):
            for family in dict.fromkeys(c['families']):
                ids=[i for i,f in enumerate(c['families']) if f==family]
                row=look[(c['panel'],c['role'],family,tuple(selected))] if selected is not None else oldlook[(c['panel'],c['role'],family,name)]
                y=np.array(row['target']);err=np.linalg.norm(pred[index,ids]-y,axis=0)/max(np.linalg.norm(y[:,0]),1e-30)
                G=c['gradient'].numpy();H=c['hessian_reference'].numpy();av=arms[index][1]
                full=-np.einsum('bop,bp->bo',G,av)-.5*np.einsum('bp,bopq,bq->bo',av,H,av)
                remainder=full[ids,0]-y[:,0];compression=pred[index,ids,0]-full[ids,0];budget=max(np.linalg.norm(y[:,0]),1e-30)
                assert np.max(abs(remainder+compression-(pred[index,ids,0]-y[:,0])))<1e-12
                decomposition.append(dict(panel=c['panel'],role=c['role'],family=family,arm=name,total=float(err[0]),full_quadratic_error=float(np.linalg.norm(remainder)/budget),compression_error=float(np.linalg.norm(compression)/budget),alignment_cosine=float(remainder@compression/max(np.linalg.norm(remainder)*np.linalg.norm(compression),1e-30))))
                scores.append(dict(panel=c['panel'],role=c['role'],family=family,arm=name,split='calibration' if c['template'] in ['near_greeted','outside_called'] else 'heldout_opened',number_error=float(err[0]),modal_error=float(max(err[1:]))))
    # Poison loaded labels; fitting interface has no labels and frozen coefficients cannot change.
    for rows in [native['records'],old['records']]:
        for row in rows:row['target']=[[float('nan')]]
    after=hashlib.sha256(b''.join(c.tobytes() for _,_,c,_ in frozen)).hexdigest();assert before==after
    summary={}
    for split in ['calibration','heldout_opened']:
        rows=[r for r in scores if r['split']==split];summary[split]=dict(number_error=max(r['number_error'] for r in rows),modal_error=max(r['modal_error'] for r in rows),passes=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows))
    out=dict(summary=summary,solver_diagnostics=solver_diagnostics,error_decomposition=decomposition,coefficient_hash_before_outcomes=before,coefficient_hash_after_poisoned_outcomes=after,checks=checks,scores=scores,literal_context_values=44,shared_plane_values=10,shared_index_values=6,scope='Frozen dictionary; analytic-response minimax fit per family group. Native outcomes loaded after all fitting. Source directions and derivatives are full native dependencies. All data opened. No fresh OOD, autonomous context generator, selectivity or composition adoption.')
    (P/'DERIVATIVE_ONLY_MINIMAX_V1_RESULT.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
