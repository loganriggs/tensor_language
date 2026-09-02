"""Canary v2: extended standing regression suite guarding the FULL final state
(supersedes bilin18_canary.py for routine use; keep both). Checks, with frozen
tolerances:
(1) original canary trio: score-rank in [3.3,5.3]; L1 linearization in
    [0.20,0.37]; dilution ratios 5->6 in [0.22,0.28] and 14->15 in [0.03,0.06];
(2) atlas integrity: all four .pt atlases load with expected component counts
    (36/24/36/24);
(3) leverage law: b18-b12 pair >= 0.70 from saved atlases;
(4) fraction law spot-check: bilin12 mlp5's direct best-match in bilin18 within
    L5-L9;
(5) depth-smoothness: bilin18 atlas smooth fraction >= 0.9."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_canary import main as canary1_main
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_canary2_results.json')

def spearman(a,b):
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std().clamp_min(1e-9)
    rb=(rb-rb.mean())/rb.std().clamp_min(1e-9)
    return float((ra*rb).mean())

def determinism_fingerprint():
    """One fixed forward; bit-level logits fingerprint appended to history.
    Time-locates any cross-session numeric flip (2026-09-02 .084-nat class:
    rung474's own code stopped reproducing its own bundle between 05:57 and
    08:02 with replay exact within-process).  DRIFT here = kernel/env change,
    NOT model damage; it dates the flip, nothing else."""
    import hashlib, os
    from bilin18_joint_removal import m, FW, DEV
    b = FW[0:4, :257].to(DEV)
    grabbed = []
    handle = m.transformer.h[17].mlp.register_forward_hook(
        lambda mod, i_, o_: grabbed.append(o_.detach()))
    try:
        with torch.no_grad():
            out = m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    finally:
        handle.remove()
    scalar = (out[0] if isinstance(out, (tuple, list)) else out).detach()
    import numpy as _np
    # fingerprint over the full layer-17 MLP write (4x256x1152) PLUS the
    # returned scalar -- bit-level over ~1.2M values, far stronger than the
    # scalar alone (the first 08:24 baseline hashed only the scalar loss)
    arr = _np.concatenate([grabbed[0].float().cpu().numpy().ravel(),
                           scalar.float().cpu().numpy().ravel()])
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "composition": "v2_layer17_mlp_plus_scalar",
             "sha": hashlib.sha256(arr.tobytes()).hexdigest(),
             "sum": float(arr.sum()), "abs_sum": float(abs(arr).sum())}
    hist = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
            'ops/determinism_fingerprint_history.jsonl')
    prev = None
    if os.path.exists(hist):
        lines = open(hist).read().strip().splitlines()
        if lines:
            prev = json.loads(lines[-1])
    if prev and prev.get("composition") != entry["composition"]:
        prev = None   # composition changed; not a drift, a new baseline
    if prev and prev["sha"] != entry["sha"]:
        print(f'DRIFT: logits fingerprint changed since {prev["ts"]} '
              f'(sum {prev["sum"]:.6f} -> {entry["sum"]:.6f}) -- '
              f'cross-session comparisons made across this boundary are unsafe')
    with open(hist, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry, bool(prev is None or prev["sha"] == entry["sha"])


def main():
    t0=time.time()
    canary1_main()   # writes its own json; DRIFT prints surface in the log
    fp_entry, fp_stable = determinism_fingerprint()
    c1=json.load(open('/workspace/tensor_language/basis_aligned/'
                      'bilinear_quotient/bilin18_canary_results.json'))
    ok1=c1['pa'] and c1['pb'] and c1['pc']
    base='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    atl={}
    counts={'bilin18_fingerprint_atlas.pt':36,'bilin12_atlas.pt':24,
            'swiglu18_atlas.pt':36,'sqrd12_atlas.pt':24}
    ok2=True
    for f,n in counts.items():
        d=torch.load(base+f)
        atl[f]=d['fingerprints']
        if len(d['fingerprints'])!=n:
            ok2=False; print(f'DRIFT: {f} has {len(d["fingerprints"])} comps')
    lev=lambda a:torch.stack([v.float().abs() for v in a.values()]).sum(0)
    r=spearman(lev(atl['bilin18_fingerprint_atlas.pt']),
               lev(atl['bilin12_atlas.pt']))
    ok3=r>=0.70
    f12=atl['bilin12_atlas.pt']['mlp5'].float()
    cur={lj:abs(spearman(f12,atl['bilin18_fingerprint_atlas.pt'][f'mlp{lj}']
                         .float())) for lj in range(18)}
    best=max(cur,key=cur.get)
    ok4=5<=best<=9
    a18=atl['bilin18_fingerprint_atlas.pt']
    keys=sorted(a18); smooth=0
    for a in keys:
        typ='mlp' if a.startswith('mlp') else 'attn'
        la=int(a[len(typ):])
        same=[b for b in keys if b.startswith(typ) and b!=a]
        best2=max(same,key=lambda b:abs(spearman(a18[a].float(),
                                                 a18[b].float())))
        if abs(int(best2[len(typ):])-la)<=2: smooth+=1
    ok5=smooth/len(keys)>=0.9
    allok=ok1 and ok2 and ok3 and ok4 and ok5
    out={'canary1':ok1,'atlases':ok2,'leverage':r,'fraction_spot':best,
         'smooth_frac':smooth/len(keys),'fingerprint': fp_entry, 'fingerprint_stable_vs_previous': fp_stable, 'ALL':bool(allok)}
    print(f'\ncanary2: c1 {"OK" if ok1 else "DRIFT"} | atlases '
          f'{"OK" if ok2 else "DRIFT"} | leverage {r:.2f} '
          f'{"OK" if ok3 else "DRIFT"} | fraction spot L{best} '
          f'{"OK" if ok4 else "DRIFT"} | smooth {smooth}/{len(keys)} '
          f'{"OK" if ok5 else "DRIFT"}')
    print(f'ALL: {"GREEN" if allok else "DRIFT DETECTED"}')
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
