"""Joint six-atom fit to analytic coefficients, with no finite-outcome fitting."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    torch.set_num_threads(2)
    previous=json.loads((P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').read_text())
    plane=np.array(previous['plane']);support=previous['supports']['shared2_plus3']
    pairs=[(i,j) for i in range(5) for j in range(i,5)]
    def pack(H):return np.stack([H[...,i,j]*(1 if i==j else 2**.5) for i,j in pairs],axis=-1)
    def unpack(z):
        H=np.zeros((*z.shape[:-1],5,5))
        for k,(i,j) in enumerate(pairs):H[...,i,j]=H[...,j,i]=z[...,k]/(1 if i==j else 2**.5)
        return H
    atoms=[np.outer(plane[:,0],plane[:,0]),(np.outer(plane[:,0],plane[:,1])+np.outer(plane[:,1],plane[:,0]))/2**.5,np.outer(plane[:,1],plane[:,1])]
    for i,j in support:
        atom=np.zeros((5,5));atom[i,j]=atom[j,i]=1/(1 if i==j else 2**.5);atoms.append(atom)
    D=pack(np.array(atoms)).T
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text())
    amplitudes=np.array([np.isin(np.arange(5),s).astype(float) for s in native['plan']['source_sets']])
    M=pack(np.einsum('bi,bj->bij',amplitudes,amplitudes))
    operators={'coefficient':np.linalg.pinv(D),'amplitude_design':np.linalg.pinv(M@D)@M}
    planted=np.random.default_rng(71).normal(size=(8,6))@D.T
    checks={mode:float(abs(planted@op.T@D.T-planted).max()) for mode,op in operators.items()}
    checks['full15']=float(abs(M@np.linalg.pinv(M)-np.eye(15)).max())
    assert max(checks.values())<1e-12
    look={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    old=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    oldlook={(r['panel'],r['role'],r['family'],r['arm']):r for r in old['records'] if r['mode']=='full'}
    cases=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases']
    scores=[];objective_checks=[]
    for c in cases:
        H=c['hessian_reference'].numpy();G=c['gradient'].numpy();flat=pack(H)
        Q=plane@plane.T;seq=np.einsum('ij,bojk,kl->boil',Q,H,Q)
        for i,j in support:seq[...,i,j]=seq[...,j,i]=H[...,i,j]
        sequential=pack(seq)
        for mode,op in operators.items():
            coeff=flat@op.T;R=unpack(coeff@D.T);metric=np.eye(15) if mode=='coefficient' else M
            oldloss=np.linalg.norm((flat-sequential)@metric.T)**2
            newloss=np.linalg.norm((flat-pack(R))@metric.T)**2
            assert newloss<=oldloss+1e-12
            objective_checks.append(dict(mode=mode,old=oldloss,new=newloss))
            arms=[('pair_design',str(s),np.broadcast_to(a,(len(G),5)),s) for s,a in zip(native['plan']['source_sets'],amplitudes)]
            arms += [('unit_null',name,a.numpy(),None) for name,a in c['amplitudes'].items()]
            for design,name,a,selected in arms:
                pred=-np.einsum('bop,bp->bo',G,a)-.5*np.einsum('bp,bopq,bq->bo',a,R,a)
                for family in dict.fromkeys(c['families']):
                    ids=[i for i,f in enumerate(c['families']) if f==family]
                    row=look[(c['panel'],c['role'],family,tuple(selected))] if selected is not None else oldlook[(c['panel'],c['role'],family,name)]
                    y=np.array(row['target']);err=np.linalg.norm(pred[ids]-y,axis=0)/max(np.linalg.norm(y[:,0]),1e-30)
                    scores.append(dict(mode=mode,split='calibration' if c['template'] in ['near_greeted','outside_called'] else 'heldout_opened',design=design,panel=c['panel'],role=c['role'],family=family,arm=name,number_error=float(err[0]),modal_error=float(max(err[1:]))))
    summary={}
    for mode in operators:
        summary[mode]={}
        for split in ['calibration','heldout_opened']:
            rows=[r for r in scores if r['mode']==mode and r['split']==split]
            summary[mode][split]=dict(number_error=max(r['number_error'] for r in rows),modal_error=max(r['modal_error'] for r in rows),passes=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows))
    out=dict(summary=summary,instrument_checks=checks,dictionary_rank=int(np.linalg.matrix_rank(D)),dictionary_condition=float(np.linalg.cond(D)),design_dictionary_condition=float(np.linalg.cond(M@D)),support=support,literal_context_values=44,shared_plane_values=10,shared_index_values=6,objective_checks=objective_checks,scope='Posthoc redteam on opened contexts. Same frozen dictionary and analytic Hessians; no native finite labels fitted. Full generators retained. Joint coefficient optimality is restricted to fixed dictionary; functional metric only uses prescribed amplitude design.',scores=scores)
    (P/'SHARED_QUADRATIC_JOINT_FIT_V1_RESULT.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['scores','objective_checks']},indent=2))
if __name__=='__main__':main()
