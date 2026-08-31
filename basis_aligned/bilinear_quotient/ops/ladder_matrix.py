"""SIMPLIFICATION LADDER x CIRCUITS (rung 155; operator's program): global simplifications scored for BREADTH.

CONVENTION (S2135): per-position dCE = CE(one component simplified) - CE(real model) on the census rows.
The operator's criterion, registered: a simplification is FAITHFUL only if it preserves MANY circuits - a
patch that rescues one circuit's datapoints is a lookup, not a mechanism. For the three biggest substrate
components (a8: 16 own circuits, a16: 13, m16: 6), a ladder of GLOBAL replacements (no circuit conditioning,
fit on census TRAIN half rows 0-499; populations per S2190): MEAN (the optimal constant; also the rung-153
protocol control), LINEAR (global affine input->output), CLSDICT (10-class constant dictionary via the
classify2 taxonomy), plus CP-2304 (weights-only) at m16. Member damage per circuit, full census; breadth
scored on TEST-half members (>= 20).

REGISTERED PREDICTIONS:
  (a) THE LADDER HELPS EVERYWHERE: for each component, median own-circuit member damage of the best
      non-constant rung <= 0.8 x the MEAN rung.
  (b) BREADTH EXISTS (the faithfulness bar): some non-constant rung of a8 keeps >= 8 of its 16 own circuits
      below 0.5 x their battery refs on test-half members.
  (c) PROTOCOL CONTROL: median MEAN-rung/rung-153-matrix ratio in [0.8, 1.25].
NULL: breadth < 3 at every rung - no global simplification of the substrate components passes many circuits;
the components are irreducibly complex at circuit grain. PRICE: linear = 1,327,104 values/comp; clsdict =
11,520/comp; CP-2304 = 7,962,624 (the ladder prices are the point). Tripwire: INSTRUMENT FAIL if any arm's
cev is bitwise equal to base. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['removal_matrix_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: simplification ladder x circuits')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ladder_matrix_results.json'

def classify2(Tk):
    import tiktoken
    enc4=tiktoken.get_encoding('gpt2')
    n=Tk.shape[0]
    Mid=torch.zeros(n,256,dtype=torch.long)
    for r in range(n):
        toks=Tk[r,:257].tolist()
        for pos in range(256):
            t=toks[pos+1]; p=toks[pos]
            tg=enc4.decode([t]); pv=enc4.decode([p]); st=tg.strip()
            if st.isdigit() and not tg.startswith(' '): k=0
            elif st in (')',']') and any(b in enc4.decode(
                toks[max(0,pos-60):pos+1]) for b in ('(','[')): k=1
            elif chr(10) in tg: k=2
            elif tg in ('.','!','?'): k=3
            elif tg==',': k=4
            elif (tg.startswith(' ') and st[:1].isupper() and
                  (pv.strip()[:1].isupper() if pv.strip() else False)): k=5
            elif t==p: k=6
            elif (not tg.startswith(' ')) and st.isalpha(): k=7
            elif t in toks[:pos+1]: k=8
            else: k=9
            Mid[r,pos]=k
    return Mid

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat(); HALF=NFLAT//2
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    R153=json.load(open(PT+'removal_matrix_results.json'))['matrix']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],
                  'top':v['mean_ablation']['top'][0]['component']}
    COMPS=['a8','a16','m16']
    LB=classify2(ROWS).reshape(-1)
    print('labels built',flush=True)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    def evalce():
        ces=[]
        for i in range(0,ROWS.shape[0],4):
            bb=ROWS[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    CAP={c:{'x':[],'y':[]} for c in COMPS}
    hs=[]
    for c in COMPS:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                CAP[c]['x'].append(i_[0].detach().reshape(-1,D).to(torch.float16).cpu())
                CAP[c]['y'].append(y.detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce()
    for h in hs: h.remove()
    FIT={}
    for c in COMPS:
        X=torch.cat(CAP[c]['x']); Y=torch.cat(CAP[c]['y'])
        MU=Y.float().mean(0).to(DEV)
        Xt=X[:HALF].float().to(DEV); Yt=Y[:HALF].float().to(DEV); Lt=LB[:HALF].to(DEV)
        lam=1e-2*Xt.shape[0]
        W=torch.linalg.solve(Xt.T@Xt+lam*torch.eye(D,device=DEV),Xt.T@Yt)
        b=Yt.mean(0)-Xt.mean(0)@W
        CV=torch.stack([Yt[Lt==k].mean(0) if int((Lt==k).sum())>0 else Yt.mean(0)
                        for k in range(10)])
        FIT[c]={'mean':MU,'linear':(W,b),'clsdict':CV}
        CAP[c]=None; del Xt,Yt
        print(f'fits done for {c}',flush=True)
    mlp16=m.transformer.h[16].mlp
    _L=mlp16.Left.weight.detach().float(); _R=mlp16.Right.weight.detach().float()
    _Dw=mlp16.Down.weight.detach().float(); _db=mlp16.Down_bias.detach().float()
    _kp=(_Dw.norm(dim=0)*_L.norm(dim=1)*_R.norm(dim=1)).argsort(descending=True)[:2304]
    CP16=(_L[_kp].contiguous(),_R[_kp].contiguous(),_Dw[:,_kp].contiguous(),_db)
    cur={'bi':0}
    def arm(c,rung):
        mod,kind=module_of(c)
        def h(mo,i_,o_):
            xin=i_[0].reshape(-1,D)
            if rung=='mean':
                new=FIT[c]['mean'].expand(xin.shape[0],D)
            elif rung=='linear':
                W,b=FIT[c]['linear']
                new=xin.float()@W+b
            elif rung=='clsdict':
                lb=LB[cur['bi']*1024:cur['bi']*1024+xin.shape[0]].to(DEV)
                new=FIT[c]['clsdict'][lb]
            else:
                Lk,Rk,Dk,db=CP16
                new=((xin.float()@Lk.T)*(xin.float()@Rk.T))@Dk.T+db
            cur['bi']=cur['bi']+1 if rung=='clsdict' else cur['bi']
            if kind=='attn':
                y,v1=o_
                return (new.view_as(y).to(y.dtype),v1)
            return new.view_as(o_).to(o_.dtype)
        hh=mod.register_forward_hook(h)
        cur['bi']=0
        cev=evalce()
        hh.remove()
        return cev
    OUTM={}
    tags=sorted(CINFO)
    for c in COMPS:
        rungs=['mean','linear','clsdict']+(['cp2304'] if c=='m16' else [])
        for rung in rungs:
            cev=arm(c,rung)
            dd=cev-CBASE
            if float(dd.abs().max())<1e-6:
                raise SystemExit(f'INSTRUMENT FAIL: {c}/{rung} bitwise equal to base')
            OUTM[f'{c}:{rung}']={t:round(float(dd[CINFO[t]['mask']].abs().mean()),4) for t in tags}
            OUTM[f'{c}:{rung}']['__agg']=round(float(dd.mean()),4)
            OUTM[f'{c}:{rung}']['__test']={t:round(float(dd[HALF:][CINFO[t]['mask'][HALF:]].abs().mean()),4)
                                           for t in tags if int(CINFO[t]['mask'][HALF:].sum())>=20}
            print(f"  {c}:{rung}: agg {OUTM[f'{c}:{rung}']['__agg']:+.4f}",flush=True)
    import statistics as stt
    pa=True; brd=0; reps=[]
    for c in COMPS:
        own=[t for t in tags if CINFO[t]['top']==c]
        mmean=stt.median([OUTM[f'{c}:mean'][t] for t in own])
        best=min(stt.median([OUTM[f'{c}:{r}'][t] for t in own])
                 for r in (['linear','clsdict']+(['cp2304'] if c=='m16' else [])))
        pa&=best<=0.8*mmean
        for t in tags:
            if c in R153: reps.append(OUTM[f'{c}:mean'][t]/max(R153[c][t],1e-9))
    for rung in ('linear','clsdict'):
        owna=[t for t in tags if CINFO[t]['top']=='a8']
        k=sum(1 for t in owna if t in OUTM[f'a8:{rung}']['__test']
              and OUTM[f'a8:{rung}']['__test'][t]<0.5*CINFO[t]['ref'])
        brd=max(brd,k)
    medrep=stt.median(reps)
    pb=brd>=8
    pc=0.8<=medrep<=1.25
    res={'matrix':OUTM,'a8_best_breadth':brd,'median_153_repro':round(medrep,3),
         'convention':'per-position dCE = CE(one component simplified) - CE(real model) on census rows',
         'pred_a_ladder_helps':bool(pa),'pred_b_breadth':bool(pb),'pred_c_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'a8 best breadth {brd}/16; 153-repro {medrep:.3f}')
    print(f"(a) ladder helps at all comps: {'HELD' if pa else 'FAILED'}")
    print(f"(b) a8 breadth {brd} >= 8: {'HELD' if pb else 'FAILED'}")
    print(f"(c) mean-rung repro {medrep:.3f} in [0.8, 1.25]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
