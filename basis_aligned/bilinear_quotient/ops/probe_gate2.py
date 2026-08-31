"""STREAM + CLASS PROBE (rung 172): does the 10-class taxonomy close the gate gap?

CONVENTION (S2135): AUC of a ridge probe on [component input stream (+) classify2 one-hot]; train rows
0-499, AUC on the test half. S2267: stream-only median AUC 0.702; class-alone (CPU check, review 1633)
median 0.581 - each weak; the question is composition. classify2 is deployable (raw tokens).

REGISTERED PREDICTIONS:
  (a) COMPOSITION HELPS: median AUC >= 0.78.
  (b) NO REGRESS: every family >= its rung-170 stream-only AUC - 0.01.
  (c) CONTROL: shuffled-train-label AUC in [0.45, 0.55] everywhere.
NULL: median <= 0.72 - the gate needs deep context beyond stream+class; the gating thread closes as
oracle-bound and per-circuit surgery stays diagnostic. PRICE: 1,162 values/gate. Tripwire: as rung 170.
Self-reviewed."""

import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['probe_gate_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: stream + class probe')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'probe_gate2_results.json'

def classify2(Tk):
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    n=Tk.shape[0]; Mid=torch.zeros(n,256,dtype=torch.long)
    for r in range(n):
        toks=Tk[r,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; p=toks[pos]
            tg=enc.decode([t]); pv=enc.decode([p]); st_=tg.strip()
            if st_.isdigit() and not tg.startswith(' '): k=0
            elif st_ in (')',']') and any(b in enc.decode(toks[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
            elif chr(10) in tg: k=2
            elif tg in ('.','!','?'): k=3
            elif tg==',': k=4
            elif (tg.startswith(' ') and st_[:1].isupper() and (pv.strip()[:1].isupper() if pv.strip() else False)): k=5
            elif t==p: k=6
            elif (not tg.startswith(' ')) and st_.isalpha(): k=7
            elif t in toks[:pos+1]: k=8
            else: k=9
            Mid[r,pos]=k
    return Mid

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
    LB2=classify2(ROWS).reshape(-1)
    OH=torch.zeros(NFLAT,10,dtype=torch.float16); OH[torch.arange(NFLAT),LB2]=1
    CAP={c:torch.cat([CAP[c],OH],dim=1) for c in CAP}
    print('inputs captured + class one-hots appended',flush=True)
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
        w=torch.linalg.solve(X.T@X+lam*torch.eye(D+10,device=DEV),X.T@(2*y-1))
        Xte=CAP[c][HALF:].float()
        sc=(Xte.to(DEV)@w).cpu()
        a=auc(sc,lab[HALF:])
        ysh=y[torch.randperm(len(y),generator=g)]
        wsh=torch.linalg.solve(X.T@X+lam*torch.eye(D+10,device=DEV),X.T@(2*ysh-1))
        ash=auc((Xte.to(DEV)@wsh).cpu(),lab[HALF:])
        rows.append({'component':c,'auc':round(a,4),'auc_shuffled':round(ash,4),'n_test_members':ntest})
        print(f'  {c}: AUC {a:.3f} (shuffled {ash:.3f}; {ntest} test members)',flush=True)
    meda=stt.median([r['auc'] for r in rows])
    a8v=[r['auc'] for r in rows if r['component']=='a8']
    pa=meda>=0.78
    R170={r['component']:r['auc'] for r in json.load(open(PT+'probe_gate_results.json'))['rows']}
    pb=all(r['auc']>=R170.get(r['component'],1.0)-0.01 for r in rows)
    pc=all(0.45<=r['auc_shuffled']<=0.55 for r in rows)
    res={'rows':rows,'median_auc':round(meda,4),
         'convention':'ridge probe on component INPUT; train rows 0-499, AUC on test half',
         'pred_a_class_helps':bool(pa),'pred_b_no_regress':bool(pb),'pred_c_shuffle_control':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median AUC {meda:.3f} >= 0.78: {'HELD' if pa else 'FAILED'}")
    print(f"(b) no family below its rung-170 AUC - 0.01: {'HELD' if pb else 'FAILED'}")
    print(f"(c) shuffle control in band: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
