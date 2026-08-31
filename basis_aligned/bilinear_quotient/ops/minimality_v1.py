"""MINIMALITY AUDIT v1 (rung 154): are the circuits one-component stories or multi-component mechanisms?

CONVENTION (S2135): per-position dCE = CE(knockout) - CE(real model) on the census rows. The 62 circuits are
census leaves + a top-component attribution - never tested for minimality. v1 measures NECESSITY DEPTH on a
stratified sample of 12 circuits (4 a8-family, 4 a16-family, 4 singleton-topped; deterministic order): member
damage under joint mean-ablation of {top1}, {top1,2}, {top1,2,3} (battery runner-up components; circuits with
fewer runners-up use what exists). Saturation = the top-1 share of the top-3 damage.

REGISTERED PREDICTIONS:
  (a) NEAR-MINIMAL AT COMPONENT GRAIN: median over sampled circuits of damage(top1)/damage(top123) >= 0.7.
  (b) BOUNDED SUPER-NECESSITY: median damage(top123)/damage(top1) <= 1.5.
  (c) PROTOCOL CONTROL: median damage(top1)/battery ref in [0.67, 1.5].
NULL: (a) < 0.5 - circuits are genuinely multi-component (the user's non-minimality concern confirmed);
their minimal sets then need the full audit. PRICE: none (attribution). Tripwire: INSTRUMENT FAIL if fewer
than 8 sampled circuits have >= 2 battery top entries. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['removal_matrix.py','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+('ops/' if f.endswith('.py') else '')+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: minimality audit v1')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'minimality_results.json'

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
        tops=[e['component'] for e in v['mean_ablation']['top'][:3]]
        CINFO[t]={'mask':mm,'ref':v['mean_ablation']['top'][0]['abs_dce_members'],'tops':tops}
    a8f=sorted(t for t,v in CINFO.items() if v['tops'][0]=='a8')[:4]
    a16f=sorted(t for t,v in CINFO.items() if v['tops'][0]=='a16')[:4]
    cnt={}
    for t,v in CINFO.items(): cnt[v['tops'][0]]=cnt.get(v['tops'][0],0)+1
    single=sorted(t for t,v in CINFO.items() if cnt[v['tops'][0]]==1)[:4]
    SAMPLE=a8f+a16f+single
    nmulti=sum(1 for t in SAMPLE if len(CINFO[t]['tops'])>=2)
    print(f'sample: {SAMPLE}; {nmulti} with >= 2 top entries',flush=True)
    if nmulti<8: raise SystemExit('INSTRUMENT FAIL: fewer than 8 sampled circuits have >= 2 top entries')
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
    comps=sorted({c for t in SAMPLE for c in CINFO[t]['tops']})
    acc={c:[torch.zeros(D,device=DEV),0] for c in comps}
    hs=[]
    for c in comps:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0)
                acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce([])
    for h in hs: h.remove()
    MU={c:acc[c][0]/max(acc[c][1],1) for c in comps}
    print(f'means captured for {len(comps)} components',flush=True)
    def ablate_set(cset):
        hs=[]
        for c in cset:
            mod,kind=module_of(c)
            def abl(mo,i_,o_,c=c,kind=kind):
                if kind=='attn':
                    y,v1=o_
                    return (MU[c].expand_as(y).to(y.dtype),v1)
                return MU[c].expand_as(o_).to(o_.dtype)
            hs.append(mod.register_forward_hook(abl))
        cev=evalce([])
        for h in hs: h.remove()
        return cev
    cache={}
    def dmg(cset,mask):
        key=tuple(sorted(cset))
        if key not in cache:
            cache[key]=ablate_set(key)
            print(f'  ablated {key}',flush=True)
        return float((cache[key]-CBASE)[mask].abs().mean())
    rows=[]
    for t in SAMPLE:
        v=CINFO[t]
        d1=dmg(v['tops'][:1],v['mask'])
        d12=dmg(v['tops'][:2],v['mask']) if len(v['tops'])>=2 else d1
        d123=dmg(v['tops'][:3],v['mask']) if len(v['tops'])>=3 else d12
        rows.append({'tag':t,'tops':v['tops'],'ref':round(v['ref'],3),
                     'd_top1':round(d1,4),'d_top12':round(d12,4),'d_top123':round(d123,4),
                     'share_top1':round(d1/max(d123,1e-9),3)})
        print(f"  {t}: top1 {d1:.3f} -> top12 {d12:.3f} -> top123 {d123:.3f} (share {rows[-1]['share_top1']:.2f})",flush=True)
    import statistics as stt
    shares=[r['share_top1'] for r in rows]
    ratios=[r['d_top123']/max(r['d_top1'],1e-9) for r in rows]
    reps=[r['d_top1']/max(r['ref'],1e-9) for r in rows]
    pa=stt.median(shares)>=0.7
    pb=stt.median(ratios)<=1.5
    pc=0.67<=stt.median(reps)<=1.5
    res={'sample':SAMPLE,'rows':rows,'median_top1_share':round(stt.median(shares),3),
         'median_top123_over_top1':round(stt.median(ratios),3),'median_battery_repro':round(stt.median(reps),3),
         'convention':'per-position dCE = CE(joint mean-ablation) - CE(real model) on census rows',
         'pred_a_near_minimal':bool(pa),'pred_b_bounded':bool(pb),'pred_c_repro':bool(pc),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median top1 share {stt.median(shares):.3f} >= 0.7: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median top123/top1 {stt.median(ratios):.3f} <= 1.5: {'HELD' if pb else 'FAILED'}")
    print(f"(c) median battery repro {stt.median(reps):.3f} in [0.67, 1.5]: {'HELD' if pc else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
