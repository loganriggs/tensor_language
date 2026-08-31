"""FAMILY CARRIERS FOR ALL 62 (rung 167): the das_subspace column in one run.

CONVENTION (S2135): share = member |dCE| of family-basis subspace patch / that circuit's full swap; census
rows. S2263: within-family sharing ratio 0.85. Here ONE pca32 basis per interchange-top component (pooled
member outputs of all that component's circuits) is scored against every circuit: 62 x {full swap, family-
subspace swap}. ~50 min run.

REGISTERED PREDICTIONS:
  (a) FAMILY BASES GENERALIZE: median share (all 62) >= 0.4.
  (b) BROAD: share >= 0.3 for >= 60% of circuits.
  (c) POPULATION GUARD: full-swap member damage within 2x the battery interchange ref for >= 50 circuits.
NULL: median < 0.25 - family bases do not generalize beyond the a8 sample; the repertoire stores per-circuit
bases after all. PRICE: a passing column = ~10 bases x 32 x 1152 = 0.37M values covering all 62 circuits.
Tripwire: INSTRUMENT FAIL on any inert arm. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['shared_carrier_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: family carriers for all 62')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'family_carriers_results.json'

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    NFLAT=CN.nflat()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    CINFO={}
    for t,v in BATC.items():
        try: lf=CN.leaf(t)
        except Exception: continue
        mm=torch.zeros(NFLAT,dtype=torch.bool); mm[lf['member']]=True
        if mm.sum()==0: continue
        CINFO[t]={'mask':mm,'itop':v['interchange']['top'][0]['component'],
                  'iref':v['interchange']['top'][0]['abs_dce_members']}
    comps=sorted({v['itop'] for v in CINFO.values()})
    print(f'{len(CINFO)} circuits over {len(comps)} interchange-top components',flush=True)
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
    CAP={c:[] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                CAP[c].append(y.detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce()
    for h in hs: h.remove()
    CAP={c:torch.cat(v) for c,v in CAP.items()}
    print('captures done',flush=True)
    FB={}
    for c in comps:
        pool=torch.cat([CINFO[t]['mask'].nonzero().squeeze(1)
                        for t in CINFO if CINFO[t]['itop']==c])
        Ym=CAP[c][pool].float()
        _,_,Vh=torch.linalg.svd((Ym-Ym.mean(0))[:20000].to(DEV),full_matrices=False)
        FB[c]=Vh[:32].T.contiguous()
    print('family bases built',flush=True)
    import statistics as stt
    rows=[]
    for i9,t in enumerate(sorted(CINFO)):
        c=CINFO[t]['itop']; mod,kind=module_of(c)
        mm=CINFO[t]['mask']
        mi=mm.nonzero().squeeze(1)
        g=torch.Generator().manual_seed(300+i9)
        perm=mi[torch.randperm(mi.numel(),generator=g)]
        SRC=torch.zeros(NFLAT,dtype=torch.long); SRC[mi]=perm
        st={'PPT':None,'bi':0}
        def hook(mo,i_,o_,c=c,kind=kind,mm=mm,SRC=SRC,st=st):
            y=o_[0] if isinstance(o_,tuple) else o_
            B9=y.reshape(-1,D).shape[0]
            lo=st['bi']*1024
            sel=mm[lo:lo+B9]
            yn=y.reshape(-1,D)
            if sel.any():
                ysrc=CAP[c][SRC[lo:lo+B9][sel]].to(DEV).float()
                ycur=yn[sel].float()
                upd=ysrc if st['PPT'] is None else ycur+(ysrc-ycur)@st['PPT']
                yn=yn.clone(); yn[sel]=upd.to(yn.dtype)
            st['bi']+=1
            yn=yn.view_as(y)
            if kind=='attn': return (yn,o_[1])
            return yn
        def arm(PPT):
            st['PPT']=PPT; st['bi']=0
            hh=mod.register_forward_hook(hook)
            cev=evalce()
            hh.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {t} inert')
            return float(d[mm].abs().mean())
        full=arm(None)
        sh=arm((FB[c]@FB[c].T).to(DEV))/full
        rows.append({'tag':t,'comp':c,'full':round(full,4),'iref':round(CINFO[t]['iref'],3),
                     'family_share':round(sh,3)})
        print(f'  {t} @ {c}: full {full:.3f} share {sh:.2f}',flush=True)
    med=stt.median([r['family_share'] for r in rows])
    nb=sum(1 for r in rows if r['family_share']>=0.3)
    pop=sum(1 for r in rows if 0.5<=r['full']/max(r['iref'],1e-9)<=2.0)
    pa=med>=0.4
    pb=nb>=0.6*len(rows)
    pc=pop>=50
    res={'rows':rows,'median_family_share':round(med,3),'n_over_0.3':nb,'pop_ok':pop,
         'convention':'share = member |dCE| of family-basis patch / full swap; census rows',
         'pred_a_generalize':bool(pa),'pred_b_broad':bool(pb),'pred_c_population':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'median family share {med:.3f}; >=0.3 for {nb}/{len(rows)}; pop_ok {pop}')
    print(f"(a) median {med:.3f} >= 0.4: {'HELD' if pa else 'FAILED'}")
    print(f"(b) {nb} >= {int(0.6*len(rows))}: {'HELD' if pb else 'FAILED'}")
    print(f"(c) population {pop} >= 50: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
