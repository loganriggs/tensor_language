"""OOD COMPONENT-GRAIN TRANSPORT (rung 176): do knockout signatures survive fresh text?

CONVENTION (S2135): per-position dCE = CE(mean-ablation) - CE(real model). Per-circuit OOD is blocked at
leaf recomputation (repertoire note); the feasible grain is COMPONENT x CLASS: each of the 16 components'
mean-ablation produces a 10-class damage profile (classify2, computable on any text). Census profiles vs
profiles on 120 FRESH pile rows (docs 5000+, unseen): if the signatures transport, the component-grain
physics is not census-specific.

REGISTERED PREDICTIONS:
  (a) SHAPE TRANSPORTS: median per-component Spearman(census profile, fresh profile) >= 0.8.
  (b) MAGNITUDE TRANSPORTS: median (fresh mean|dCE| / census mean|dCE|) in [0.6, 1.6].
  (c) GUARD: fresh base CE in [2.0, 5.0] nats AND no inert arm.
NULL: rho < 0.5 - knockout signatures are census-specific; the OOD column records non-transport.
PRICE: none (the OOD column is the product). Tripwire: INSTRUMENT FAIL on inert arms. Self-reviewed."""
import json, sys, time, os
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
if os.environ.get('BQLIB_DRYRUN')=='1':
    _bq='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
    _need=['minset_splice2_results.json','removal_matrix_results.json','circuits/BATTERY.json']
    _miss=[f for f in _need if not os.path.exists(_bq+f)]
    if _miss:
        print(f'DRYRUN FAIL: missing {_miss}'); raise SystemExit(1)
    print('DRYRUN OK: OOD component-grain transport')
    raise SystemExit(0)
import torch
import torch.nn.functional as F
from bilin18_joint_removal import m, DEV
D=1152
PT='/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT=PT+'ood_transport_results.json'

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

def main():
    T00=time.time()
    sys.path.insert(0,'/workspace/rspd')
    import census_lib as CN
    CN.use_state('census_state_diverse.pt')
    ROWS=CN.rows().cpu()
    CBASE=CN.base_ce().float().cpu()
    BATC=json.load(open(PT+'circuits/BATTERY.json'))['by_tag']
    SIX=sorted({v['mean_ablation']['top'][0]['component'] for v in BATC.values()})
    import tiktoken
    from datasets import load_dataset
    enc3=tiktoken.get_encoding('gpt2')
    dsf=load_dataset('NeelNanda/pile-10k',split='train')
    frows=[]
    for di in range(5000,10000):
        tk=enc3.encode_ordinary(dsf[di]['text'])
        for st0 in range(0,len(tk)-513,513):
            frows.append(tk[st0:st0+513]); break
        if len(frows)>=120: break
    FRESH=torch.tensor(frows,dtype=torch.long)
    def module_of(c):
        li=int(c[1:])
        return (m.transformer.h[li].attn,'attn') if c[0]=='a' else (m.transformer.h[li].mlp,'mlp')
    def evalce(tok):
        ces=[]
        for i in range(0,tok.shape[0],4):
            bb=tok[i:i+4,:257].to(DEV)
            idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
            with torch.no_grad():
                x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
                for blk in m.transformer.h: x,v1=blk(x,v1,x0)
                lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
            ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,reduction='none').cpu())
        return torch.cat(ces)
    FBASE=evalce(FRESH)
    fb=float(FBASE.mean())
    print(f'fresh base CE {fb:.3f}',flush=True)
    LBc=classify2(ROWS).reshape(-1); LBf=classify2(FRESH).reshape(-1)
    acc={c:[torch.zeros(D,device=DEV),0] for c in SIX}
    hs=[]
    for c in SIX:
        mod,kind=module_of(c)
        def mk(c=c):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                acc[c][0]+=y.detach().reshape(-1,D).float().sum(0); acc[c][1]+=y.reshape(-1,D).shape[0]
            return h
        hs.append(mod.register_forward_hook(mk()))
    _=evalce(ROWS)
    for h in hs: h.remove()
    MU={c:acc[c][0]/max(acc[c][1],1) for c in SIX}
    import statistics as stt
    def srho(u,v):
        a=torch.tensor(u).argsort().argsort().float(); b=torch.tensor(v).argsort().argsort().float()
        a=a-a.mean(); b=b-b.mean()
        return float((a*b).sum()/((a.norm()*b.norm())+1e-9))
    rows=[]
    for c in SIX:
        mod,kind=module_of(c)
        def abl(mo,i_,o_,c=c,kind=kind):
            if kind=='attn':
                y,v1=o_
                return (MU[c].expand_as(y).to(y.dtype),v1)
            return MU[c].expand_as(o_).to(o_.dtype)
        hh=mod.register_forward_hook(abl)
        dc=evalce(ROWS)-CBASE
        df=evalce(FRESH)-FBASE
        hh.remove()
        if float(dc.abs().max())<1e-6 or float(df.abs().max())<1e-6:
            raise SystemExit(f'INSTRUMENT FAIL: {c} inert')
        pc=[float(dc[LBc==k].mean()) if int((LBc==k).sum())>0 else 0.0 for k in range(10)]
        pf=[float(df[LBf==k].mean()) if int((LBf==k).sum())>0 else 0.0 for k in range(10)]
        rows.append({'component':c,'rho':round(srho(pc,pf),3),
                     'mag_ratio':round(float(df.abs().mean())/max(float(dc.abs().mean()),1e-9),3),
                     'census_profile':[round(x,3) for x in pc],'fresh_profile':[round(x,3) for x in pf]})
        print(f"  {c}: rho {rows[-1]['rho']:.2f} mag {rows[-1]['mag_ratio']:.2f}",flush=True)
    medr=stt.median([r['rho'] for r in rows])
    medm=stt.median([r['mag_ratio'] for r in rows])
    pa=medr>=0.8
    pb=0.6<=medm<=1.6
    pc9=2.0<=fb<=5.0
    res={'rows':rows,'median_rho':round(medr,3),'median_mag_ratio':round(medm,3),'fresh_base_ce':round(fb,3),
         'convention':'per-position dCE = CE(mean-ablation) - CE(real model); 10-class profiles, census vs 120 fresh rows',
         'pred_a_shape':bool(pa),'pred_b_magnitude':bool(pb),'pred_c_guard':bool(pc9),
         'self_reviewed':True,'runtime_s':round(time.time()-T00,1)}
    json.dump(res,open(OUT,'w'),indent=1)
    print(f"(a) median rho {medr:.3f} >= 0.8: {'HELD' if pa else 'FAILED'}")
    print(f"(b) median mag ratio {medm:.3f} in [0.6, 1.6]: {'HELD' if pb else 'FAILED'}")
    print(f"(c) fresh base {fb:.3f} in [2, 5]: {'HELD' if pc9 else 'FAILED'}")
    print(f'wrote {OUT} ({time.time()-T00:.0f}s)')

if __name__=='__main__':
    main()
