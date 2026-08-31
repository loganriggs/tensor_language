"""DAS-LITE (rung 161; C3 of the circuits program, S2255): fixed low-rank counterfactual subspaces.

CONVENTION (S2135): per-position dCE = CE(subspace interchange) - CE(real model) on the census rows. For the
ten rung-160 circuits, at each interchange-top component: candidate subspaces are FIXED (no learning) - top-r
PCA of member outputs (r in {1, 8, 32}) and the rank-1 member-vs-offslice difference-in-means. Patched arm:
member positions receive y + P P^T (y_src - y) with the SAME seeded member-permutation sources as rung 160.
share(P) = member |dCE| of the subspace patch / member |dCE| of rung 160's full swap. The rung-160 receipt is
read at RUN time (runner order guarantees it; absence is INSTRUMENT FAIL). This is the cheap forerunner of
DAS proper (learned rank-r rotations, Geiger et al.) - if fixed subspaces already carry the effect, learning
starts warm; if not, the learned search is necessary.

REGISTERED PREDICTIONS:
  (a) LOW-RANK CARRIERS EXIST: median share(PCA r=32) >= 0.5.
  (b) VERY-LOW-RANK: share(PCA r=8) >= 0.4 for >= 3 of 10 circuits.
  (c) INSTRUMENT: full-swap reproduction for 2 circuits within 10% of rung 160, AND share monotone in r
      (tol 0.05) for >= 8 of 10.
NULL: shares < 0.25 at r=32 - the causal variable is distributed across the full component output; only
DAS-proper (learned rotations) can find a carrier, or none exists. PRICE: none (instrument; a passing
subspace costs r x 1152 values in the repertoire). Tripwire: INSTRUMENT FAIL if the rung-160 receipt is
missing or any patched cev is bitwise equal to base. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['ops/interchange_inst.py','circuits/BATTERY.json','circuits/REPERTOIRE.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: DAS-lite fixed subspaces')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'daslite_results.json'

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
        SUBS={'pca1':Vh9[:1].T.contiguous(),'pca8':Vh9[:8].T.contiguous(),
              'pca32':Vh9[:32].T.contiguous(),'dmean1':dm.unsqueeze(1)}
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
        if len(rows9)<2:
            hh=mod.register_forward_hook(ih)
            cur9['bi']=0
            cev=evalce([])
            hh.remove()
            mdf=float((cev-CBASE)[ISM].abs().mean())
            ent['full_repro']=round(mdf,4)
            print(f"  {t} full-swap repro {mdf:.3f} vs 160's {fullmd:.3f}",flush=True)
        rows9.append(ent)
    med32=stt.median([r['shares']['pca32'] for r in rows9])
    n8=sum(1 for r in rows9 if r['shares']['pca8']>=0.4)
    mono=sum(1 for r in rows9 if r['shares']['pca1']<=r['shares']['pca8']+0.05
             and r['shares']['pca8']<=r['shares']['pca32']+0.05)
    reprook=all(abs(r['full_repro']-r['full_member_160'])<=0.1*max(r['full_member_160'],1e-9)
                for r in rows9 if 'full_repro' in r)
    pa=med32>=0.5
    pb=n8>=3
    pc=reprook and mono>=8
    res={'rows':rows9,'median_share_pca32':round(med32,3),'n_pca8_over_0.4':n8,'monotone_count':mono,
         'convention':'share = member |dCE| of subspace patch / rung-160 full swap; census rows',
         'pred_a_lowrank_exists':bool(pa),'pred_b_very_lowrank':bool(pb),'pred_c_instrument':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f'median share(r=32) {med32:.3f}; pca8>=0.4 count {n8}; monotone {mono}/10; repro {reprook}')
    print(f"(a) median share(32) {med32:.3f} >= 0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b) pca8 >= 0.4 for {n8} >= 3: {'HELD' if pb else 'FAILED'}")
    print(f"(c) repro + monotone {mono} >= 8: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')
    return

if __name__=='__main__':
    main()
