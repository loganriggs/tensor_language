"""SHARED-CARRIER TEST (rung 166): is the a8 family's counterfactual carrier ONE subspace?

CONVENTION (S2135): share = member |dCE| of subspace patch / that circuit's full swap; census rows. S2258
found rank-32 carriers per circuit; the compression question (and the S2248/S2218 substrate story) predicts
they are SHARED within a component family. Reference basis: pca32 of the a8-family circuit r.2.0's member
outputs (the rung-160/161 sample). Targets: the four next-largest a8-family circuits. Arms per target: full
swap, OWN pca32 (its own members' basis), REF pca32 (r.2.0's basis).

REGISTERED PREDICTIONS:
  (a) SUBSTANTIALLY SHARED: median over targets of [ref-share / own-share] >= 0.7.
  (b) ABSOLUTELY USEFUL: ref-share >= 0.3 for >= 3 of 4 targets.
  (c) INSTRUMENT: full-swap member damage within 2x of the battery mean-ablation ref at every target
      (population guard), and no inert arm.
NULL: ref-share < 0.2 - carriers are circuit-specific even within a family; the repertoire stores 62
subspaces, not a few. PRICE: none (instrument; a shared carrier would store ONE 32 x 1152 basis for 16
circuits). Tripwire: INSTRUMENT FAIL on inert arms. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['daslite_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: shared-carrier test')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'shared_carrier_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    fam=[]
    for t,v in BATC.items():
        if v['mean_ablation']['top'][0]['component']!='a8': continue
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        fam.append((t,mm,v['mean_ablation']['top'][0]['abs_dce_members']))
    fam.sort(key=lambda x:-int(x[1].sum()))
    REF=[x for x in fam if x[0]=='r.2.0'][0]
    TGT=[x for x in fam if x[0]!='r.2.0'][:4]
    print(f"ref r.2.0; targets {[x[0] for x in TGT]}",flush=True)
    mod=m.transformer.h[8].attn
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
    CAP=[]
    hh=mod.register_forward_hook(lambda mo,i_,o_: CAP.append(o_[0].detach().reshape(-1,D).to(torch.float16).cpu()))
    _=evalce()
    hh.remove()
    Y=torch.cat(CAP)
    def basis32(mm):
        Ym=Y[mm.nonzero().squeeze(1)].float()
        _,_,Vh=torch.linalg.svd((Ym-Ym.mean(0))[:20000].to(DEV),full_matrices=False)
        return Vh[:32].T.contiguous()
    PREF=basis32(REF[1])
    import statistics as stt
    rows=[]
    for i9,(t,mm,ref) in enumerate(TGT):
        mi=mm.nonzero().squeeze(1)
        g=torch.Generator().manual_seed(700+i9)
        perm=mi[torch.randperm(mi.numel(),generator=g)]
        SRC=torch.zeros(NFLAT,dtype=torch.long); SRC[mi]=perm
        POWN=basis32(mm)
        st={'PPT':None,'bi':0}
        def hook(mo,i_,o_):
            y=o_[0]
            B9=y.reshape(-1,D).shape[0]
            lo=st['bi']*1024
            sel=mm[lo:lo+B9]
            yn=y.reshape(-1,D)
            if sel.any():
                ysrc=Y[SRC[lo:lo+B9][sel]].to(DEV).float()
                ycur=yn[sel].float()
                upd=ysrc if st['PPT'] is None else ycur+(ysrc-ycur)@st['PPT']
                yn=yn.clone(); yn[sel]=upd.to(yn.dtype)
            st['bi']+=1
            return (yn.view_as(y),o_[1])
        def arm(PPT):
            st['PPT']=PPT; st['bi']=0
            hh=mod.register_forward_hook(hook)
            cev=evalce()
            hh.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {t} inert arm')
            return float(d[mm].abs().mean())
        full=arm(None)
        so=arm((POWN@POWN.T).to(DEV))/full
        sr=arm((PREF@PREF.T).to(DEV))/full
        rows.append({'tag':t,'full':round(full,4),'battery_ref':round(ref,3),
                     'own_share':round(so,3),'ref_share':round(sr,3),
                     'ratio':round(sr/max(so,1e-9),3)})
        print(f"  {t}: full {full:.3f} | own {so:.2f} | ref {sr:.2f} (ratio {rows[-1]['ratio']:.2f})",flush=True)
    medr=stt.median([r['ratio'] for r in rows])
    nabs=sum(1 for r in rows if r['ref_share']>=0.3)
    popok=all(0.5<=r['full']/max(r['battery_ref'],1e-9)<=2.0 for r in rows)
    pa=medr>=0.7
    pb=nabs>=3
    pc=popok
    res={'rows':rows,'median_ref_over_own':round(medr,3),'n_refshare_over_0.3':nabs,
         'convention':'share = member |dCE| of subspace patch / full swap; ref basis = r.2.0 pca32',
         'pred_a_shared':bool(pa),'pred_b_useful':bool(pb),'pred_c_population':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'median ref/own {medr:.3f}; ref>=0.3 for {nabs}/4')
    print(f"(a) median ratio {medr:.3f} >= 0.7: {'HELD' if pa else 'FAILED'}")
    print(f"(b) ref-share >= 0.3 for {nabs} >= 3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) population guard: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
