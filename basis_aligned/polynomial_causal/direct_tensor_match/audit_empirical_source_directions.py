"""Decode exported programs independently and audit all registered fit predictions.

Also report centered private-branch removal and private quadratic-form agreement
between restarts. These are descriptive: no new semantic or OOD claim.
"""
from pathlib import Path
import json
import torch
from compact_source_graph import source_reads as execute

P = Path(__file__).parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)

def main():
    meta = json.loads((P / 'EMPIRICAL_SOURCE_DIRECTIONS_V1.json').read_text())
    programs = torch.load(P / 'EMPIRICAL_SOURCE_DIRECTIONS_PROGRAMS_V1.pt', weights_only=True)
    d = torch.load(P / 'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt', weights_only=True)
    root = torch.linalg.inv(d['inverse_root'])
    ids = d['indices']; z = d['z'][ids]; h = d['h'][ids]
    scale = (h.square().mean(-1) + torch.finfo(torch.float32).eps).sqrt()[:, None]
    ar = torch.stack([pair['a'] for pair in d['pairs']], 1)
    alpha = torch.stack([pair['alpha'] for pair in d['pairs']])
    beta = torch.stack([pair['beta'] for pair in d['pairs']])
    truth = torch.stack([pair['truth'][ids] for pair in d['pairs']], 1)
    denom = (truth - truth.mean(0)).norm(dim=0)
    def phi(q):
        return ((h @ ar - .5*q[:, ::2])/scale-alpha)*(q[:, 1::2]/scale-beta)
    rows=[]; private_forms={}
    for rec in meta['records']:
        p=programs[rec['key']]
        L,R,W,V,v=[p[n] for n in ('left_reader','right_reader','product_weights','square_reader','square_weights')]
        raw=torch.einsum('ir,ro,jr->oij',L,W,R)
        Q=(raw+raw.transpose(-1,-2))/2
        private=(V*v)@V.T; Q[5]+=private
        dense=torch.einsum('ni,oij,nj->no',z,Q,z)+z@p['source_linear']+p['source_bias']
        direct=execute(z,p)
        execution=float((dense-direct).norm()/dense.norm())
        hat=torch.einsum('ai,oij,jb->oab',root,Q,root)/d['scales'][:,None,None]
        coefficient=float((hat-d['teacher']).norm()/d['teacher'].norm())
        values=((phi(dense)-truth).norm(dim=0)/denom).tolist()
        replay=max(execution,abs(coefficient-rec['original_coefficient_error']),max(abs(a-b) for a,b in zip(values,rec['per_mode_errors'])))
        floats=sum(a.numel() for a in p.values() if a.is_floating_point())
        products=L.shape[1]+V.shape[1]
        assert replay<1e-8 and floats==896198 and products==399
        # Remove only the centered quadratic part, preserving the mean/affine convention.
        delta=z-d['mu']
        removed=dense.clone()
        removed[:,5]-=(delta@V).square()@v-torch.trace(d['old_covariance']@private)
        without=((phi(removed)-truth).norm(dim=0)/denom).tolist()
        unchanged=float((phi(removed)[:,:2]-phi(dense)[:,:2]).abs().max())
        assert unchanged==0
        private_forms[rec['key']]=(root@private@root/d['scales'][5]).flatten()
        rows.append(dict(key=rec['key'],lam=rec['lam'],replay_error=replay,
                         source_products=products,stored_floats=floats,
                         original_coefficient_error=coefficient,per_mode_errors=values,
                         without_centered_private_errors=without,
                         third_error_reduction_from_private=1-values[2]/without[2],
                         other_components_max_change=unchanged))
    selected={lam:next(r for r in rows if r['key']==key) for lam,key in meta['winners'].items()}
    primary,control=selected['1'],selected['0']
    predictions=dict(pred_a_instrument=all(r['replay_error']<1e-8 for r in rows),
        pred_b_values=all(x<=.15 and x<=1.1*y for x,y in zip(primary['per_mode_errors'],meta['plan']['scalar_baseline'])),
        pred_c_coefficient=primary['original_coefficient_error']<=1.1*control['original_coefficient_error'])
    assert predictions==meta['predictions']
    stability={}
    for lam in meta['plan']['lambdas']:
        a,b=[private_forms[r['key']] for r in rows if r['lam']==lam]
        stability[str(lam)]=dict(cosine=float((a@b)/(a.norm()*b.norm())),relative_difference=float((a-b).norm()/a.norm()))
    out=dict(rows=rows,predictions=predictions,private_restart_comparison=stability,
             scope='Opened 448 states; centered branch removal is algebraic selectivity, not fresh semantic validation. Native earlier and later inputs remain supplied.')
    (P/'EMPIRICAL_SOURCE_DIRECTIONS_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    main()
