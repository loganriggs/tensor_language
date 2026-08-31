"""PER-LEAF PROBES (rung 175): was the union the problem?

CONVENTION (S2135): AUC of a ridge probe on the component INPUT predicting a SINGLE leaf's membership;
train rows 0-499, AUC on the test half. S2267/S2269 probed FAMILY unions (many leaves mixed - a union of
distinct context patterns can be linearly inseparable even when each leaf is separable). Ten largest leaves,
probed individually at their top component's input.

REGISTERED PREDICTIONS:
  (a) LEAVES ARE CHEAP: median per-leaf AUC >= 0.85.
  (b) BROAD: AUC >= 0.75 for >= 8 of 10 leaves.
  (c) CONTROL: shuffled-train-label AUC in [0.45, 0.55] everywhere.
NULL: median < 0.7 - gating is oracle-bound at EVERY grain; the S2269 closure is final. PRICE: 1,152
values/leaf gate. Tripwire: INSTRUMENT FAIL if a leaf has < 200 test members. Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['probe_gate2_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: per-leaf probes')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'probe_leaf_results.json'

def auc(scores,labels):
    order=scores.argsort()
    ranks=torch.empty_like(order,dtype=torch.float); ranks[order]=torch.arange(len(order),dtype=torch.float)
    pos=labels.bool()
    n1=int(pos.sum()); n0=len(labels)-n1
    if n1==0 or n0==0: return float('nan')
    return float((ranks[pos].sum()-n1*(n1-1)/2)/(n1*n0))

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    NFLAT=CN.nflat(); HALF=NFLAT//2
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    leafsets={}
    for t,v in BATC.items():
        c=v['mean_ablation']['top'][0]['component']
        try: lf=CN.leaf(t)
        except Exception: continue
        leafsets[t]=(c,torch.as_tensor(lf['member'],dtype=torch.long))
    picks=sorted(leafsets,key=lambda t:-leafsets[t][1].numel())[:10]
    FAMS={t:leafsets[t][1] for t in picks}
    COMP={t:leafsets[t][0] for t in picks}
    print('leaves:',picks,flush=True)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    CAP={c:[] for c in FAMS}
    hs=[]
    comps9=sorted(set(COMP.values()))
    CAPC={c:[] for c in comps9}
    for c in comps9:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                CAPC[c].append(i_[0].detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mk()))
    with torch.no_grad():
        for i in range(0,ROWS.shape[0],4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    CAPC={c:torch.cat(v) for c,v in CAPC.items()}
    CAP={t:CAPC[COMP[t]] for t in FAMS}
    print('inputs captured',flush=True)
    import statistics as stt
    rows=[]
    g=torch.Generator().manual_seed(11)
    for c in sorted(FAMS):
        lab=torch.zeros(NFLAT,dtype=torch.bool); lab[FAMS[c]]=True
        ntest=int(lab[HALF:].sum())
        if ntest<200: raise SystemExit(f'INSTRUMENT FAIL: leaf {c} test members {ntest} < 200')
        mtr=lab[:HALF].nonzero().squeeze(1)
        ntr=(~lab[:HALF]).nonzero().squeeze(1)
        ntr=ntr[torch.randperm(ntr.numel(),generator=g)[:mtr.numel()]]
        tri=torch.cat([mtr,ntr])
        X=CAP[c][tri].float().to(DEV)
        y=lab[tri].float().to(DEV)
        lam=1e-2*len(X)
        w=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),X.T@(2*y-1))
        Xte=CAP[c][HALF:].float()
        sc=(Xte.to(DEV)@w).cpu()
        a=auc(sc,lab[HALF:])
        ysh=y[torch.randperm(len(y),generator=g)]
        wsh=torch.linalg.solve(X.T@X+lam*torch.eye(D,device=DEV),X.T@(2*ysh-1))
        ash=auc((Xte.to(DEV)@wsh).cpu(),lab[HALF:])
        rows.append({'component':c,'leaf':c,'auc':round(a,4),'auc_shuffled':round(ash,4),'n_test_members':ntest})
        print(f'  {c}: AUC {a:.3f} (shuffled {ash:.3f}; {ntest} test members)',flush=True)
    meda=stt.median([r['auc'] for r in rows])
    pa=meda>=0.85
    pb=sum(1 for r in rows if r['auc']>=0.75)>=8
    pc=all(0.45<=r['auc_shuffled']<=0.55 for r in rows)
    res={'rows':rows,'median_auc':round(meda,4),
         'convention':'ridge probe on component INPUT; train rows 0-499, AUC on test half',
         'pred_a_leaves_cheap':bool(pa),'pred_b_broad':bool(pb),'pred_c_shuffle_control':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median AUC {meda:.3f} >= 0.85: {'HELD' if pa else 'FAILED'}")
    print(f"(b) AUC >= 0.75 for {sum(1 for r in rows if r['auc']>=0.75)} of 10 >= 8: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffle control in band: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
