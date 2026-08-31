"""SUFFICIENCY (rung 171): the missing half of minimality - do the top-5 components SUFFICE?

CONVENTION (S2135): per-position dCE = CE(keep-set real, other measured components mean-ablated) - CE(real
model) on the census rows. Necessity is done (median k* = 5, S2260); sufficiency was never run. For 10
stratified circuits (4 a8-family, 3 a16-family, 3 singleton-topped): keep the circuit's battery top-5
components real and mean-ablate the REST of the 16-component measured set (scope stated: the 16 components,
not all 68 modules). Controls per circuit: keep-RANDOM-5 (seeded) and keep-NONE (all 16 ablated).

REGISTERED PREDICTIONS:
  (a) TOP-5 SUFFICES WITHIN SCOPE: median member damage(keep-top5) <= 1.0 x that circuit's battery ref.
  (b) SPECIFICITY: member damage(keep-top5) <= 0.5 x damage(keep-random5) for >= 7 of 10.
  (c) KEEPING HELPS: damage(keep-top5) <= 0.7 x damage(keep-none) for >= 8 of 10.
NULL: members die regardless of what is kept (circuits need the whole 16-component substrate - the
substrate-wide reading). PRICE: none (attribution). Tripwire: INSTRUMENT FAIL on inert arms. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['carrier_null_results.json','minimality6_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: sufficiency (keep-top5)')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'sufficiency_results.json'

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
                  'tops':[e['component'] for e in v['mean_ablation']['top'][:5]],
                  'top':v['mean_ablation']['top'][0]['component']}
    SIXTEEN=sorted({v['top'] for v in CINFO.values()})
    a8f=sorted(t for t,v in CINFO.items() if v['top']=='a8')[:4]
    a16f=sorted(t for t,v in CINFO.items() if v['top']=='a16')[:3]
    cnt={}
    for t,v in CINFO.items(): cnt[v['top']]=cnt.get(v['top'],0)+1
    single=sorted(t for t,v in CINFO.items() if cnt[v['top']]==1)[:3]
    SAMPLE=a8f+a16f+single
    print(f'sample {SAMPLE}; 16-set {SIXTEEN}',flush=True)
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
    acc={c:[torch.zeros(D,device=DEV),0] for c in SIXTEEN}
    hs=[]
    for c in SIXTEEN:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0); acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce()
    for h in hs: h.remove()
    MU={c:acc[c][0]/max(acc[c][1],1) for c in SIXTEEN}
    cache={}
    def dmg(ablset,mask):
        key=tuple(sorted(ablset))
        if key not in cache:
            hs=[]
            for c in key:
                mod,kind=module_of(c)
                def abl(mo,i_,o_,c=c,kind=kind):
                    if kind=='attn':
                        y,v1=o_
                        return (MU[c].expand_as(y).to(y.dtype),v1)
                    return MU[c].expand_as(o_).to(o_.dtype)
                hs.append(mod.register_forward_hook(abl))
            cev=evalce()
            for h in hs: h.remove()
            d=cev-CBASE
            if float(d.abs().max())<1e-6: raise SystemExit(f'INSTRUMENT FAIL: {key} inert')
            cache[key]=d
        return float(cache[key][mask].abs().mean())
    import statistics as stt
    g=torch.Generator().manual_seed(21)
    rows=[]
    for t in SAMPLE:
        v=CINFO[t]
        keep=set(v['tops'])&set(SIXTEEN)
        abl_top=[c for c in SIXTEEN if c not in keep]
        rk=[SIXTEEN[i] for i in torch.randperm(16,generator=g)[:5].tolist()]
        abl_rand=[c for c in SIXTEEN if c not in rk]
        d5=dmg(abl_top,v['mask'])
        dr=dmg(abl_rand,v['mask'])
        d0=dmg(SIXTEEN,v['mask'])
        rows.append({'tag':t,'keep':sorted(keep),'ref':round(v['ref'],3),
                     'd_keeptop5':round(d5,4),'d_keeprand5':round(dr,4),'d_keepnone':round(d0,4)})
        print(f"  {t}: keep-top5 {d5:.3f} | keep-rand5 {dr:.3f} | keep-none {d0:.3f} (ref {v['ref']:.2f})",flush=True)
    pa=stt.median([r['d_keeptop5']/max(r['ref'],1e-9) for r in rows])<=1.0
    pb=sum(1 for r in rows if r['d_keeptop5']<=0.5*r['d_keeprand5'])>=7
    pc=sum(1 for r in rows if r['d_keeptop5']<=0.7*r['d_keepnone'])>=8
    res={'rows':rows,
         'median_top5_over_ref':round(stt.median([r['d_keeptop5']/max(r['ref'],1e-9) for r in rows]),3),
         'convention':'per-position dCE = CE(keep-set real, rest of 16 mean-ablated) - CE(real model); census rows',
         'pred_a_suffices':bool(pa),'pred_b_specific':bool(pb),'pred_c_keeping_helps':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median top5/ref {res['median_top5_over_ref']:.3f} <= 1.0: {'HELD' if pa else 'FAILED'}")
    print(f"(b) top5 <= 0.5 x rand5 for {sum(1 for r in rows if r['d_keeptop5']<=0.5*r['d_keeprand5'])} >= 7: {'HELD' if pb else 'FAILED'}")
    print(f"(c) top5 <= 0.7 x none for {sum(1 for r in rows if r['d_keeptop5']<=0.7*r['d_keepnone'])} >= 8: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
