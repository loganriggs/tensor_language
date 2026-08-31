"""THE VARIANCE-VS-CAUSAL NULL BATTERY (rung 169; math review 1607): what exactly is the carrier?

CONVENTION (S2135): share = member |dCE| of subspace patch / rung-160 full swap; census rows. S2262 showed
distributed-but-concentrated (~29x per-dim); S2264 showed the subspace adds nothing to removal selectivity.
The missing control decides what the carrier IS. Three rank-32 bases per circuit at the same component:
MEMBER-PCA (the S2258 construction), OFFSLICE-PCA (principal variance of the SAME component at non-member
positions - variance structure without member specificity), RANDOM (seeded QR - dimension counting). Each
basis's captured member-variance fraction is recorded (the damage-vs-variance spectral question rides free).

REGISTERED PREDICTIONS:
  (a) CAUSAL BEYOND VARIANCE: median share(member-pca) >= 1.5 x median share(offslice-pca).
  (b) VARIANCE BEATS DIMENSION COUNTING: median share(offslice-pca) >= 2 x median share(random).
  (c) INSTRUMENT: 2-circuit full-swap repro within 10% of rung 160 AND median share(random) <= 0.3.
NULL: member ~ offslice - the "carrier" is just the component's principal variance, and the counterfactual-
circuit reading demotes to "variance patching" (an honest downgrade the repertoire would record). PRICE:
none. Tripwire: INSTRUMENT FAIL on inert arms or missing rung-160 receipt. Self-reviewed."""


import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['carrier_removal_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: variance-vs-causal null battery')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'carrier_null_results.json'

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
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],
                  'top':v['mean_ablation']['top'][0]['component']}
    for t,v in BATC.items():
        if t in CINFO:
            CINFO[t]['iref']=v['interchange']['top'][0]['abs_dce_members']
            CINFO[t]['itop']=v['interchange']['top'][0]['component']
    SAMPLE=sorted(CINFO,key=lambda t:-int(CINFO[t]['mask'].sum()))
    seen=set(); PICK=[]
    for t in SAMPLE:
        if CINFO[t]['itop'] in seen: continue
        seen.add(CINFO[t]['itop']); PICK.append(t)
        if len(PICK)==10: break
    print(f'picked {PICK}',flush=True)
    comps=sorted({CINFO[t]['itop'] for t in PICK})
    print(f'{len(CINFO)} circuits; {len(comps)} distinct top components: {comps}',flush=True)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    def evalce(hooks):
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
    # component census means
    MU={}
    hs=[]; acc={}
    for c in comps:
        mod,kind=module_of(c)
        acc[c]=[torch.zeros(D,device=DEV),0]
        def mk(c=c,kind=kind):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0)
                acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce([])
    for h in hs: h.remove()
    for c in comps: MU[c]=(acc[c][0]/max(acc[c][1],1))
    print('component means captured',flush=True)
    CAPOUT={c:[] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mkc(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                CAPOUT[c].append(y.detach().reshape(-1,D).to(torch.float16).cpu())
            return h
        hs.append(mod.register_forward_hook(mkc()))
    _=evalce([])
    for h in hs: h.remove()
    CAPOUT={c:torch.cat(v) for c,v in CAPOUT.items()}
    print('outputs captured',flush=True)
    p160=PT+'interchange_inst_results.json'
    if not os.path.exists(p160): raise SystemExit('INSTRUMENT FAIL: rung-160 receipt missing at run time')
    R160={r['tag']:r for r in json.load(open(p160))['rows']}
    g9=torch.Generator().manual_seed(77)
    import statistics as stt
    rows9=[]
    for t in PICK:
        c=CINFO[t]['itop']; mod,kind=module_of(c)
        mi=CINFO[t]['mask'].nonzero().squeeze(1)
        perm=mi[torch.randperm(mi.numel(),generator=g9)]
        SRC=torch.zeros(NFLAT,dtype=torch.long); SRC[mi]=perm
        ISM=CINFO[t]['mask']
        cur9={'bi':0}
        def ih(mo,i_,o_,c=c,kind=kind):
            y=o_[0] if isinstance(o_,tuple) else o_
            B9=y.reshape(-1,D).shape[0]
            lo=cur9['bi']*1024
            sel=ISM[lo:lo+B9]
            yn=y.reshape(-1,D).clone()
            if sel.any():
                yn[sel]=CAPOUT[c][SRC[lo:lo+B9][sel]].to(y.device).float().to(yn.dtype)
            cur9['bi']+=1
            yn=yn.view_as(y)
            if kind=='attn': return (yn,o_[1])
            return yn
        Ymem=CAPOUT[c][mi].float()
        Ymu=Ymem.mean(0)
        _,_,Vh9=torch.linalg.svd((Ymem-Ymu)[:20000].to(DEV),full_matrices=False)
        offsl=(~ISM).nonzero().squeeze(1)
        offsl=offsl[torch.randperm(offsl.numel(),generator=g9)[:20000]]
        dm=(Ymem.mean(0)-CAPOUT[c][offsl].float().mean(0)).to(DEV)
        dm=dm/dm.norm().clamp_min(1e-9)
        offsl2=(~ISM).nonzero().squeeze(1)
        offsl2=offsl2[torch.randperm(offsl2.numel(),generator=g9)[:20000]]
        Yoff=CAPOUT[c][offsl2].float()
        _,_,VhO=torch.linalg.svd((Yoff-Yoff.mean(0)).to(DEV),full_matrices=False)
        gR=torch.Generator().manual_seed(4242+len(rows9))
        QR,_=torch.linalg.qr(torch.randn(D,32,generator=gR).to(DEV))
        SUBS={'mempca32':Vh9[:32].T.contiguous(),'offpca32':VhO[:32].T.contiguous(),
              'rand32':QR[:,:32].contiguous()}
        Ymc=(Ymem-Ymu)[:20000].to(DEV)
        _tot=float(Ymc.pow(2).sum())
        VARF={k:round(float((Ymc@P).pow(2).sum())/max(_tot,1e-9),3) for k,P in SUBS.items()}
        fullmd=R160[t]['member_absdce']
        ent={'tag':t,'comp':c,'full_member_160':fullmd,'shares':{}}
        for sname,P in SUBS.items():
            PPT=(P@P.T).to(DEV)
            def ihs(mo,i_,o_,c=c,kind=kind,PPT=PPT):
                y=o_[0] if isinstance(o_,tuple) else o_
                B9=y.reshape(-1,D).shape[0]
                lo=cur9['bi']*1024
                sel=ISM[lo:lo+B9]
                yn=y.reshape(-1,D).clone()
                if sel.any():
                    ysrc=CAPOUT[c][SRC[lo:lo+B9][sel]].to(y.device).float()
                    ycur=yn[sel].float()
                    yn[sel]=(ycur+(ysrc-ycur)@PPT).to(yn.dtype)
                cur9['bi']+=1
                yn=yn.view_as(y)
                if kind=='attn': return (yn,o_[1])
                return yn
            hh=mod.register_forward_hook(ihs)
            cur9['bi']=0
            cev=evalce([])
            hh.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6:
                raise SystemExit(f'INSTRUMENT FAIL: {t}/{sname} bitwise equal to base')
            md=float(d[ISM].abs().mean())
            ent['shares'][sname]=round(md/max(fullmd,1e-9),3)
            print(f"  {t} @ {c} [{sname}]: member {md:.3f} share {ent['shares'][sname]:.2f}",flush=True)
        ent['varfrac']=VARF
        if len(rows9)<2:
            hh=mod.register_forward_hook(ih)
            cur9['bi']=0
            cev=evalce([])
            hh.remove()
            mdf=float((cev-CBASE)[ISM].abs().mean())
            ent['full_repro']=round(mdf,4)
            print(f"  {t} full-swap repro {mdf:.3f} vs 160's {fullmd:.3f}",flush=True)
        rows9.append(ent)
    medM=stt.median([r['shares']['mempca32'] for r in rows9])
    medO=stt.median([r['shares']['offpca32'] for r in rows9])
    medR=stt.median([r['shares']['rand32'] for r in rows9])
    reprook=all(abs(r['full_repro']-r['full_member_160'])<=0.1*max(r['full_member_160'],1e-9)
                for r in rows9 if 'full_repro' in r)
    pa=medM>=1.5*medO
    pb=medO>=2*medR
    pc=reprook and medR<=0.3
    res={'rows':rows9,'median_mem':round(medM,3),'median_off':round(medO,3),'median_rand':round(medR,3),
         'convention':'share = member |dCE| of subspace patch / rung-160 full swap; three-way null battery',
         'pred_a_causal_beyond_variance':bool(pa),'pred_b_variance_beats_random':bool(pb),'pred_c_instrument':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'medians: member-pca {medM:.3f} | offslice-pca {medO:.3f} | random {medR:.3f}; repro {reprook}')
    print(f"(a) member {medM:.3f} >= 1.5 x offslice ({1.5*medO:.3f}): {'HELD' if pa else 'FAILED'}")
    print(f"(b) offslice {medO:.3f} >= 2 x random ({2*medR:.3f}): {'HELD' if pb else 'FAILED'}")
    print(f"(c) repro + random <= 0.3: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')
    return

if __name__=='__main__':
    main()
