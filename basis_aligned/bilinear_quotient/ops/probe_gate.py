"""MEMBER-PROBE SEPARABILITY (rung 170): is the position gate cheaply computable?

CONVENTION (S2135): AUC of a ridge probe on the component INPUT stream predicting circuit-family membership;
census rows; train = rows 0-499, AUC on the test half. S2264/S2266: circuit identity is positional, and all
surgical selectivity comes from the member gate. Deployability hinges on whether that gate is a cheap
function of the local stream. Families: the six components with >= 4 own circuits (a8, a16, a3, m14, m13,
m16); label = union of the family's circuits' members.

REGISTERED PREDICTIONS:
  (a) GATES ARE CHEAP: median family AUC >= 0.85.
  (b) a8 AUC >= 0.90.
  (c) CONTROL: shuffled-train-label AUC in [0.45, 0.55] at every family.
NULL: median AUC < 0.7 - the gate needs the oracle; deployable selective removal/extraction stays blocked.
PRICE: a passing gate costs 1,152 values (one probe vector) per family. Tripwire: INSTRUMENT FAIL if any
family has < 500 test members. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['carrier_null_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: member-probe separability')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'probe_gate_results.json'

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
    famsets={}
    for t,v in BATC.items():
        c=v['mean_ablation']['top'][0]['component']
        try: lf=CN.leaf(t)
        except Exception: continue
        famsets.setdefault(c,[]).append(torch.as_tensor(lf['member'],dtype=torch.long))
    FAMS={c:torch.unique(torch.cat(v)) for c,v in famsets.items() if len(v)>=4}
    print('families:',sorted(FAMS),flush=True)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    CAP={c:[] for c in FAMS}
    hs=[]
    for c in FAMS:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                CAP[c].append(i_[0].detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mk()))
    with torch.no_grad():
        for i in range(0,ROWS.shape[0],4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous()
            x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
            for blk in m.transformer.h: x,v1=blk(x,v1,x0)
    for h in hs: h.remove()
    CAP={c:torch.cat(v) for c,v in CAP.items()}
    print('inputs captured',flush=True)
    import statistics as stt
    rows=[]
    g=torch.Generator().manual_seed(11)
    for c in sorted(FAMS):
        lab=torch.zeros(NFLAT,dtype=torch.bool); lab[FAMS[c]]=True
        ntest=int(lab[HALF:].sum())
        if ntest<500: raise SystemExit(f'INSTRUMENT FAIL: family {c} test members {ntest} < 500')
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
        rows.append({'component':c,'auc':round(a,4),'auc_shuffled':round(ash,4),'n_test_members':ntest})
        print(f'  {c}: AUC {a:.3f} (shuffled {ash:.3f}; {ntest} test members)',flush=True)
    meda=stt.median([r['auc'] for r in rows])
    a8v=[r['auc'] for r in rows if r['component']=='a8']
    pa=meda>=0.85
    pb=bool(a8v and a8v[0]>=0.90)
    pc=all(0.45<=r['auc_shuffled']<=0.55 for r in rows)
    res={'rows':rows,'median_auc':round(meda,4),
         'convention':'ridge probe on component INPUT; train rows 0-499, AUC on test half',
         'pred_a_gates_cheap':bool(pa),'pred_b_a8':bool(pb),'pred_c_shuffle_control':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median AUC {meda:.3f} >= 0.85: {'HELD' if pa else 'FAILED'}")
    print(f"(b) a8 AUC {a8v[0] if a8v else float('nan'):.3f} >= 0.90: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffle control in band: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
