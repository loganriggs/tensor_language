"""Minimum-norm near-optimal coefficients: test an optimizer-choice confound."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,nnls
from shared_selective_source_lp import choose
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/refined_selective_sources_v1_result.json'
a=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.])[a['parent_mapping']];groups=[];certificates=[];records=[]
for fit in a['gradients']:
    if fit['panel']!='opposite':continue
    weights=[]
    for g in np.asarray(fit['gradient']):
        start,ceiling=choose(g,reference);target=float(-g[0]@reference);scale=max(abs(target),1e-10)
        n=-g[0]*np.sign(target)/scale;control=g[1:]/scale;gamma=.99*ceiling['retention']
        matrix=np.r_[-n[None],control,-control,np.eye(23),-np.eye(23)];rhs=np.r_[-gamma,np.full(16,.08),np.ones(46)]
        result=minimize(lambda x:.5*x@x,start,jac=lambda x:x,constraints=[dict(type='ineq',fun=lambda x:rhs-matrix@x,jac=lambda x:-matrix)],method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
        x=result.x;slack=rhs-matrix@x;active=np.flatnonzero(slack<1e-7)
        multipliers,_=nnls(matrix[active].T,-x,maxiter=10000)
        stationarity=float(np.linalg.norm(x+matrix[active].T@multipliers));dual=-.5*np.linalg.norm(matrix[active].T@multipliers)**2-multipliers@rhs[active]
        gap=float(.5*x@x-dual);violation=float(max(0.,-slack.min()))
        valid=violation<=1e-7 and abs(gap)<=1e-6 and stationarity<=1e-6
        certificates.append(dict(valid=bool(valid),solver_success=bool(result.success),violation=violation,duality_gap=gap,stationarity=stationarity,retention=gamma,original_norm=float(np.linalg.norm(start)),canonical_norm=float(np.linalg.norm(x))))
        weights.append(x)
    weights=np.array(weights);groups.append(dict(role=fit['role'],template=fit['template'],weights=weights.tolist()))
    held=next(g for g in a['gradients'] if g['panel']=='congruent' and g['role']==fit['role'] and g['template']==fit['template']);g=np.asarray(held['gradient']);families=held['families'];baseline=-g[:,0]@reference
    for mode,aa in [('same_lexical_pair',weights),('next_noun_same_number',np.roll(weights,-2,axis=0))]:
        pred=-np.einsum('boi,bi->bo',g,aa)
        for family in dict.fromkeys(families):
            ids=[i for i,f in enumerate(families) if f==family];n=pred[ids,0];b=baseline[ids];ret=float(n@b/(b@b));leak=float(np.linalg.norm(pred[ids,1:],axis=0).max()/max(np.linalg.norm(n),1e-30))
            records.append(dict(role=fit['role'],family=family,mode=mode,retention=ret,control_ratio=leak,passed=bool(ret>=.8 and leak<=.1)))
valid=all(c['valid'] for c in certificates)
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),valid=valid,certificates=certificates,groups=groups,records=records,summary={mode:sum(r['passed'] for r in records if r['mode']==mode) for mode in ['same_lexical_pair','next_noun_same_number']},scope='Fit-only minimum-norm choice at99%ofper-input LP ceiling. Strictly convex objective with numerical primal/dual checks. Held derivative tests only; no native validation or independent extraction.')
(P/'CANONICAL_SOURCE_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(valid=valid,summary=out['summary'],worst_gap=max(abs(c['duality_gap']) for c in certificates),worst_stationarity=max(c['stationarity'] for c in certificates))))
