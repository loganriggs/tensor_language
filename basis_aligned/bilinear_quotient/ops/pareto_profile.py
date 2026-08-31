"""PARETO CIRCUIT-PROFILE INVARIANCE (rung 177; CPU): do the two frontier configs damage the SAME circuits?

CONVENTION (S2135): member abs-dCE per circuit from the two registered Pareto receipts (rung 132: 47.8M ->
+1.9474; rung 137: 63.7M -> +1.7202). S2218 predicts near-identical profiles (shared vulnerability).

REGISTERED PREDICTIONS:
  (a) INVARIANCE: Spearman(profile_132, profile_137) >= 0.95.
  (b) UNIFORM RELIEF: median per-circuit damage ratio (137/132) in [0.7, 1.0].
  (c) COVERAGE: both receipts carry 62 circuits.
NULL: rho < 0.8 - Pareto points damage different circuits and config choice matters at circuit grain.
PRICE: none (CPU on receipts). Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['frontier_rows_results.json','frontier_claim4608_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: Pareto profile invariance (CPU)')
    raise SystemExit(0)
import torch, statistics as stt, time as _t
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'pareto_profile_results.json'
T00=_t.time()
A={r['tag']:r['member_absdce'] for r in json.load(open(PT+'frontier_rows_results.json'))['circuits']}
B={r['tag']:r['member_absdce'] for r in json.load(open(PT+'frontier_claim4608_results.json'))['circuits']}
tags=sorted(set(A)&set(B))
u=[A[t] for t in tags]; v=[B[t] for t in tags]
a=torch.tensor(u).argsort().argsort().float(); b=torch.tensor(v).argsort().argsort().float()
a=a-a.mean(); b=b-b.mean()
rho=float((a*b).sum()/((a.norm()*b.norm())+1e-9))
ratios=[B[t]/max(A[t],1e-9) for t in tags]
medr=stt.median(ratios)
pa=rho>=0.95
pb=0.7<=medr<=1.0
pc=len(A)==62 and len(B)==62
res={'n':len(tags),'spearman':round(rho,4),'median_ratio_137_over_132':round(medr,4),
     'pred_a_invariant':bool(pa),'pred_b_uniform_relief':bool(pb),'pred_c_coverage':bool(pc),
     'self_reviewed':True,'runtime_s':round(_t.time()-T00,2)}
json.dump(res,open(OUT,'w'),indent=1)
print(f'spearman {rho:.4f}; median ratio {medr:.3f}; n {len(tags)}')
print(f"(a) rho {rho:.4f} >= 0.95: {'HELD' if pa else 'FAILED'}")
print(f"(b) median ratio {medr:.3f} in [0.7, 1.0]: {'HELD' if pb else 'FAILED'}")
print(f"(c) coverage 62/62: {'HELD' if pc else 'FAILED'}")
print(f'wrote {OUT}')
