"""Optimization-independent necessary error bounds for every cross-output pair.

A three-mode rank-one tensor has rank-one every unfolding. For each pair, the
smaller nonzero squared singular value of an unfolding lower-bounds CP-rank-one
approximation error. Compute it from two-column Gram determinants; take the
maximum of all three lower bounds. Strongest bound is not generally achievable.
"""
from pathlib import Path
import json,torch
from shared_product_merge import fit_pair_cores
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
out=p/'MIDPOINT_SHARED_PRODUCT_MERGE_BOUND_V1.json';assert not out.exists()
e=torch.load(p/'MIDPOINT_PRIVATE_FROZEN_V1.pt',weights_only=True)['private512'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();n-=n.mean(0);m-=m.mean(0)
S=torch.load(p/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].double()
factors=[n@e['A']/len(n)**.5,m@e['B']/len(m)**.5,S@e['reduced_writers']]
grams=[v.T@v for v in factors];scales=[g.diag().clamp_min(1e-30).sqrt() for g in grams];corr=[g/s[:,None]/s[None,:] for g,s in zip(grams,scales)]
amplitudes=scales[0]*scales[1]*scales[2]
ids=torch.triu_indices(512,512,1);cross=corr[2][ids[0],ids[1]].abs()<.999999;ids=ids[:,cross].T
c=torch.stack([g[ids[:,0],ids[:,1]].clamp(-1,1) for g in corr],1)
a,b=amplitudes[ids].unbind(1);energy=a*a+b*b+2*a*b*c.prod(1);bounds=[]
for mode in range(3):
 other=[k for k in range(3) if k!=mode]
 det=a*a*b*b*(1-c[:,mode].square())*(1-c[:,other].prod(1).square())
 # Rationalized root avoids cancellation when the smaller singular value is tiny.
 discr=(energy.square()-4*det).clamp_min(0).sqrt()
 bounds.append(2*det/(energy+discr).clamp_min(1e-30))
bound=torch.stack(bounds,1).max(1).values
ratio=bound/torch.minimum(a*a,b*b)
# Validate closed form against explicit tiny tensor SVDs on a fixed subset.
subset=torch.arange(min(128,len(ids)))
ii=ids[subset];pg=torch.stack([g[ii[:,:,None],ii[:,None,:]] for g in corr],1)
fit=fit_pair_cores(pg,amplitudes[ii]);t=fit['core'];direct=[]
for mode in range(3):
 matrix=t.movedim(mode+1,1).reshape(len(t),2,4)
 sv=torch.linalg.svdvals(matrix);direct.append(sv[:,1].square())
actual=torch.stack(direct,1).max(1).values
replay=float((actual-bound[subset]).abs().max()/energy[subset].max());assert replay<1e-12
best=ratio.argsort()[:16];ii=ids[best];pg=torch.stack([g[ii[:,:,None],ii[:,None,:]] for g in corr],1)
refit=fit_pair_cores(pg,amplitudes[ii],restarts=32,steps=100,seed=261310)
upper=refit['error']/torch.minimum(a[best].square(),b[best].square())
assert bool((upper+1e-9>=ratio[best]).all())
result=dict(cross_output_pairs=len(ids),threshold=.5,pairs_not_ruled_out=int((ratio<=.5+1e-10).sum()),best_lower_bound_vs_deletion=float(ratio.min()),bound_svd_relative_replay_error=replay,stronger_fits=[dict(pair=ids[j].tolist(),lower=float(ratio[j]),achieved_upper=float(upper[k])) for k,j in enumerate(best.tolist())],conclusion='If pairs_not_ruled_out is zero, no pair-to-one rank-one edit can meet the registered threshold in this fixed separable metric, including pairs omitted by the initial cosine screen. This is not a lower bound for multi-product edits, changed upstream features, native behavior, or an arbitrary DAG.')
out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
