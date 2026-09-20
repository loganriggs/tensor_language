"""Shared quadratic dictionary from coefficient tensors; held-out construction split."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
PAIRS=((0,0),(0,1),(0,2),(1,1),(1,2),(2,2));SCALE=np.array([1,2**.5,2**.5,1,2**.5,1])
def unpack(z):
    h=np.zeros(z.shape[:-1]+(3,3))
    for k,(i,j) in enumerate(PAIRS):h[...,i,j]=h[...,j,i]=z[...,k]/SCALE[k]
    return h
def main():
    path=P/'SOURCE_AMPLITUDE_QUADRATIC_V1.json';artifact=json.loads(path.read_text());keys=list(artifact['contexts']);coeff=np.array([artifact['contexts'][k] for k in keys]);z=coeff[:,3:]*SCALE
    train=np.array([any('/'+t+'|' in k for t in ['near_greeted','outside_called']) for k in keys]);assert sum(train)==sum(~train)==96
    _,sv,vh=np.linalg.svd(z[train],full_matrices=False);basis=vh[:2]
    sparse=basis.copy()
    for row in sparse:row[np.argsort(abs(row))[:-3]]=0
    rng=np.random.default_rng(20260920);random=np.linalg.qr(rng.normal(size=(6,2)))[0].T
    ht=unpack(z[train]);eig,vec=np.linalg.eigh(np.einsum('cik,cjk->ij',ht,ht));input_basis=vec[:,-2:];proj=input_basis@input_basis.T
    zh={name:z@np.linalg.pinv(b)@b for name,b in [('shared2',basis),('random2',random),('sparse_shared2',sparse)]}
    ih=proj[None]@unpack(z)@proj[None]
    zh['input_rank2']=np.stack([ih[:,i,j]*SCALE[k] for k,(i,j) in enumerate(PAIRS)],axis=-1)
    zh['full']=z
    assert np.max(abs(z-z@vh.T@vh))<1e-12
    # Frobenius isometry catches off-diagonal weighting errors.
    assert np.max(abs(np.sum(z*z,axis=1)-np.sum(unpack(z)**2,axis=(1,2))))<1e-12
    lookup={k:i for i,k in enumerate(keys)};source=json.loads((A/'semantic_source_jet_v2r1_result.json').read_text());scores=[]
    for c in source['records']:
        ids=[lookup[f"{c['panel']}/{c['role']}/{c['family']}/{i}"] for i in range(len(c['target']))];amp=np.array(source['plan']['amplitudes'][c['arm']],dtype=float);target=np.array(c['target']);den=max(np.linalg.norm(target),1e-30)
        errors={}
        for name,values in zh.items():
            h=unpack(values[ids]);pred=-coeff[ids,:3]@amp-.5*np.einsum('i,bij,j->b',amp,h,amp)
            errors[name]=float(np.linalg.norm(pred-target)/den)
        scores.append(dict(panel=c['panel'],role=c['role'],family=c['family'],arm=c['arm'],split='train' if train[ids[0]] else 'heldout',errors=errors))
    worst={split:{arm:{method:max(c['errors'][method] for c in scores if c['split']==split and c['arm']==arm) for method in zh} for arm in source['plan']['amplitudes']} for split in ['train','heldout']}
    eligible=[c for c in scores if c['split']=='heldout' and c['arm']!='double'];a=True;b=all(c['errors']['shared2']<=.1 for c in eligible);d=max(c['errors']['shared2'] for c in eligible)<max(c['errors']['random2'] for c in eligible)
    result=dict(predictions=dict(pred_a_instrument=a,pred_b_shared2_prediction=b,pred_c_beats_matched_random=d),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_contexts=int(sum(train)),heldout_contexts=int(sum(~train)),coefficient_singular_values=sv.tolist(),dictionary=basis.tolist(),sparse_dictionary=sparse.tolist(),random_dictionary=random.tolist(),input_basis=input_basis.tolist(),price_values=dict(full=dict(shared=0,per_context=9),shared2=dict(shared=12,per_context=5),random2=dict(shared=12,per_context=5),sparse_shared2=dict(shared=6,sparse_indices=6,per_context=5),input_rank2=dict(shared=6,per_context=6)),worst_errors=worst,scores=scores,scope='Already opened texts; construction-held-out coefficient basis only. Per-context mixing coefficients still require native Hessians. No fresh OOD or native capability promotion.')
    (P/'SHARED_SOURCE_QUADRATICS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'predictions':result['predictions'],'worst_errors':worst},indent=2))
if __name__=='__main__':main()
