"""Test one common rank-two source feature plane, with matched controls."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    torch.set_num_threads(2)
    cases=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)['cases']
    train=[c for c in cases if c['template'] in ['near_greeted','outside_called']]
    def frame(cs):
        H=np.concatenate([c['hessian_reference'].numpy().reshape(-1,5,5) for c in cs])
        gram=np.einsum('cij,ckj->ik',H,H);w,v=np.linalg.eigh(gram)
        return v[:,-2:],w
    plane,eigen=frame(train)
    frames={'shared2':plane,'full5':np.eye(5)}
    training_H=np.concatenate([c['hessian_reference'].numpy() for c in train])
    output_gram=np.einsum('boij,bokj->oik',training_H,training_H)
    _,output_vectors=np.linalg.eigh(output_gram)
    frames['output_specific2']=output_vectors[...,-2:]
    for seed in [17,29,43]:frames[f'random2_{seed}']=np.linalg.qr(np.random.default_rng(seed).normal(size=(5,2)))[0]
    pairs=[(i,j) for i in range(5) for j in range(i,5)]
    projection=plane@plane.T
    residual=training_H-np.einsum('ij,bojk,kl->boil',projection,training_H,projection)
    energy=np.array([np.sum(residual[...,i,j]**2)*(1 if i==j else 2) for i,j in pairs])
    chosen=np.argsort(-energy)[:3]
    supports={'shared2_plus3':[pairs[k] for k in chosen], 'shared2_plus15':pairs}
    for seed in [17,29,43]:supports[f'shared2_random3_{seed}']=[pairs[k] for k in np.random.default_rng(seed).choice(15,3,replace=False)]
    for name in supports:frames[name]=plane
    left,_=frame([c for c in train if c['template']=='near_greeted']);right,_=frame([c for c in train if c['template']=='outside_called'])
    overlap=np.linalg.svd(left.T@right,compute_uv=False)
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text())
    look={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    old=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    oldlook={(r['panel'],r['role'],r['family'],r['arm']):r for r in old['records'] if r['mode']=='full'}
    scores=[];closure=[]
    for c in cases:
        H=c['hessian_reference'].numpy();G=c['gradient'].numpy();split='calibration' if c['template'] in ['near_greeted','outside_called'] else 'heldout_opened'
        for mode,F in frames.items():
            if F.ndim==2:
                core=np.einsum('ip,boij,jq->bopq',F,H,F);R=np.einsum('ip,bopq,jq->boij',F,core,F)
            else:
                core=np.einsum('oip,boij,ojq->bopq',F,H,F);R=np.einsum('oip,bopq,ojq->boij',F,core,F)
            if mode in supports:
                for i,j in supports[mode]:R[...,i,j]=H[...,i,j];R[...,j,i]=H[...,j,i]
            if mode in ['full5','shared2_plus15']:closure.append(float(abs(R-H).max()))
            arms=[('pair_design',str(s),np.broadcast_to(np.isin(np.arange(5),s).astype(float),(len(G),5)),s) for s in native['plan']['source_sets']]
            arms += [('unit_null',name,a.numpy(),None) for name,a in c['amplitudes'].items()]
            for design,name,a,selected in arms:
                pred=-np.einsum('bop,bp->bo',G,a)-.5*np.einsum('bp,bopq,bq->bo',a,R,a)
                for family in dict.fromkeys(c['families']):
                    ids=[i for i,f in enumerate(c['families']) if f==family]
                    row=look[(c['panel'],c['role'],family,tuple(selected))] if selected is not None else oldlook[(c['panel'],c['role'],family,name)]
                    y=np.array(row['target']);err=np.linalg.norm(pred[ids]-y,axis=0)/max(np.linalg.norm(y[:,0]),1e-30)
                    scores.append(dict(mode=mode,split=split,design=design,panel=c['panel'],role=c['role'],family=family,arm=name,number_error=float(err[0]),modal_error=float(max(err[1:]))))
    assert max(closure)<1e-12
    summary={}
    for mode in frames:
        summary[mode]={}
        for split in ['calibration','heldout_opened']:
            rows=[r for r in scores if r['mode']==mode and r['split']==split]
            summary[mode][split]=dict(number_error=max(r['number_error'] for r in rows),modal_error=max(r['modal_error'] for r in rows),passes=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows))
    out=dict(summary=summary,supports=supports,calibration_residual_pair_energy=energy.tolist(),sparse_context_values=44,sparse_shared_index_values=6,plane=plane.tolist(),input_gram_eigenvalues=eigen.tolist(),calibration_template_principal_cosines=overlap.tolist(),full_recovery=max(closure),literal_shared_values=10,output_specific_shared_values=40,literal_context_values=32,scope='One shared rank2 linear source plane, dense symmetric2x2 output cores and full five-coordinate gradients. Frozen from calibration coefficient tensors; heldout constructions already opened. Same native derivative/context costs. Stability measures subspaces, not individually identified features.',scores=scores)
    (P/'SHARED_FIVE_SOURCE_SPARSE_RESIDUAL_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='scores'},indent=2))
if __name__=='__main__':main()
