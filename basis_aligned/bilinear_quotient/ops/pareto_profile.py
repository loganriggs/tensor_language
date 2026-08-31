"""PARETO/ERA CIRCUIT-PROFILE INVARIANCE (rung 177 v2; CPU): did the grammar change move WHICH circuits pay?

CONVENTION (S2135): member abs-dCE per circuit from two registered receipts with full rows: rung 99 (the
TABLE-era S2144 config, census +2.8553) and rung 132 (the CP-era frontier, +1.9474). v1 targeted the rung-137
receipt, which stores no circuit rows (discovered at build time; recorded) - the era comparison is the
stronger question anyway: S2218 predicts the damage PROFILE is invariant even across the table->CP grammar
change.

REGISTERED PREDICTIONS:
  (a) INVARIANCE ACROSS ERAS: Spearman(profile_99, profile_132) >= 0.9.
  (b) BROAD RELIEF: median per-circuit damage ratio (132/99) in [0.4, 0.85].
  (c) COVERAGE: both receipts carry 62 circuits.
NULL: rho < 0.7 - the grammar change moved which circuits pay; circuit costs are config-specific.
PRICE: none (CPU on receipts). Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['frontier_rows_results.json','frontier_certificate_results.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: era profile invariance (CPU)')
    raise SystemExit(0)
import torch, statistics as stt, time as _t
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'pareto_profile_results.json'

def main():
    T00=_t.time()
    A={r['tag']:r['member_absdce'] for r in json.load(open(PT+'frontier_certificate_results.json'))['circuits']}
    B={r['tag']:r['member_absdce'] for r in json.load(open(PT+'frontier_rows_results.json'))['circuits']}
    tags=sorted(set(A)&set(B))
    ra=torch.tensor([A[t] for t in tags]).argsort().argsort().float()
    rb=torch.tensor([B[t] for t in tags]).argsort().argsort().float()
    ra=ra-ra.mean(); rb=rb-rb.mean()
    rho=float((ra*rb).sum()/((ra.norm()*rb.norm())+1e-9))
    ratios=[B[t]/max(A[t],1e-9) for t in tags]
    medr=stt.median(ratios)
    pa=rho>=0.9
    pb=0.4<=medr<=0.85
    pc=len(A)==62 and len(B)==62
    res={'n':len(tags),'spearman':round(rho,4),'median_ratio_132_over_99':round(medr,4),
         'pred_a_invariant':bool(pa),'pred_b_broad_relief':bool(pb),'pred_c_coverage':bool(pc),
         'self_reviewed':True,'runtime_s':round(_t.time()-T00,2)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'spearman {rho:.4f}; median ratio {medr:.3f}; n {len(tags)}')
    print(f"(a) rho {rho:.4f} >= 0.9: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median ratio {medr:.3f} in [0.4, 0.85]: {'HELD' if pb else 'FAILED'}")
    print(f"(c) coverage 62/62: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT}')

if __name__=='__main__':
    main()
