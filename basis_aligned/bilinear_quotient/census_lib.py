"""census_lib -- stable API for circuit work on bilin18.

Purpose: swarm agents call these functions instead of writing
transform scripts (the dominant failure mode: anchor-mismatch
crashes). Everything here is ported from scripts whose results are
in BILIN18_CONNECTION.md; behavior is frozen -- extend by adding
functions, never by editing semantics in place.

Conventions:
- "census grid": the 212x256 fit-window token grid, flat length
  54272. Global position gi -> (row gi//256, pos gi%256).
- leaf probes: ('pca', comp, slice_tag, (s0,s1)) = project out that
  slice-conditioned output-PCA block of component comp.
- All CE vectors are per-position nats on the census grid unless a
  tok argument says otherwise.
"""
import json, os, torch
import torch.nn.functional as F
sys_path='/workspace/tensor_language/basis_aligned/bilinear_quotient'
import sys
if sys_path not in sys.path: sys.path.insert(0,sys_path)
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152; T=256
PT=sys_path+'/'
CIRC=PT+'circuits/'

_st=None
STATE_PATH=PT+'census_state.pt'
def state():
    global _st
    if _st is None:
        _st=torch.load(STATE_PATH,map_location='cpu',
                       weights_only=False)
        _st['by_tag']={lf['tag']:lf for lf in _st['leaves']}
    return _st

def use_state(path):
    """Switch to another census state file (e.g. the diverse
    tree). Additive extension (2026-08-20): resets every cached
    singleton so all downstream functions read the new grid."""
    global _st,STATE_PATH,_FEAT,_MUS,_OUTCAP
    STATE_PATH=path; _st=None; _FEAT=None; _MUS=None
    _OUTCAP={}

def nflat():
    return rows().shape[0]*256

def leaf(tag): return state()['by_tag'][tag]
def all_tags(): return [lf['tag'] for lf in state()['leaves']]
def rows(): return state()['rows']
def base_ce(): return state()['basev'].float()

_enc=None
def enc():
    global _enc
    if _enc is None:
        import tiktoken; _enc=tiktoken.get_encoding('gpt2')
    return _enc
def d1(t):
    t=int(t)
    return '<BOS>' if (t<0 or t>50256) else enc().decode([t])
def context(gi,back=12,fwd_=0):
    gi=int(gi); r_,p_=gi//256,gi%256
    toks=rows()[r_].tolist()
    pre=enc().decode(toks[max(0,p_-back):p_+1])
    post=enc().decode(toks[p_+1:p_+1+fwd_]) if fwd_ else ''
    return pre,d1(toks[p_+1]),post

MODS={f'a{li}':m.transformer.h[li].attn for li in range(18)}
MODS.update({f'm{li}':m.transformer.h[li].mlp for li in range(18)})

_OUTCAP={}
@torch.no_grad()
def capture_out(key):
    """Component output over the census grid (54272 x D, fp16 cpu)."""
    if key in _OUTCAP: return _OUTCAP[key]
    R=rows(); capsX=[]
    def cap(mo,i_,o_):
        y=o_[0] if isinstance(o_,tuple) else o_
        capsX.append(y.detach().half().reshape(-1,D).cpu())
    h=MODS[key].register_forward_hook(cap)
    for i in range(0,R.shape[0],4):
        bb=R[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    h.remove()
    _OUTCAP[key]=torch.cat(capsX)
    return _OUTCAP[key]

_PCA={}
@torch.no_grad()
def pca_block(key,stag,blk):
    kk=(key,stag,tuple(blk))
    if kk in _PCA: return _PCA[kk]
    sl=leaf(stag)['member'] if stag in state()['by_tag'] \
        else torch.arange(nflat())
    Y=capture_out(key)[sl].float().to(DEV)
    _,_,Vh=torch.linalg.svd((Y-Y.mean(0))[:20000],full_matrices=False)
    _PCA[kk]=Vh[blk[0]:blk[1]]
    return _PCA[kk]

def leaf_hooks(probes):
    """Hooks reproducing the census's own operator for EVERY probe
    kind: pca -> projection removal; comp -> whole-component mean;
    head -> head z zeroed (the census's historical head op -- kept
    for leaf-fidelity even though zeroing is otherwise deprecated,
    see LESSONS 1)."""
    probes=[eval(p) if isinstance(p,str) else p for p in probes]
    hs=[]
    pcas=[p for p in probes if p[0]=='pca']
    if pcas: hs+=proj_hooks(pcas)
    comps=[p[1] for p in probes if p[0]=='comp']
    if comps: hs+=mean_hooks(comps)
    for p in probes:
        if p[0]=='head':
            li,hd=p[1],p[2]
            at=m.transformer.h[li].attn
            def fh(mo_,args,out,li=li,hd=hd,at=at):
                y,v1r=out
                X=args[0]; v1=args[1] if args[1] is not None else v1r
                z,vm=head_parts(li,X,v1)
                z[:,hd]=0
                Bb,Tq=X.shape[0],X.shape[1]
                yn=at.c_proj(z.transpose(1,2).contiguous()
                             .view(Bb,Tq,-1).to(X.dtype))
                return (yn,v1r)
            hs.append(at.register_forward_hook(fh))
    return hs

LAST_PROJ_RANK={}


def proj_hooks(probes):
    """probes: list of ('pca',key,stag,(s0,s1)) (strings also ok)."""
    probes=[eval(p) if isinstance(p,str) else p for p in probes]
    # 2026-08-20 (wave-7 reviewer catch, r.23.2.3): a bundle can
    # list overlapping spans from the same cached slice -- (4,16)
    # alongside (4,10) and (10,16) -- and orth() is a plain QR with
    # no rank truncation, so the projector came out rank 28 while
    # the record's component table implied 16. Identical (key,stag,
    # span) entries are dropped, spans nested inside another span
    # of the same slice are dropped, and the true rank is recorded
    # for the caller to report.
    seen=set(); keep=[]
    for pr in probes:
        _,key,stag,blk=pr
        if (key,stag,tuple(blk)) in seen: continue
        seen.add((key,stag,tuple(blk))); keep.append(pr)
    drop=set()
    for i,(_,k1,s1,b1) in enumerate(keep):
        for j,(_,k2,s2,b2) in enumerate(keep):
            if i==j or k1!=k2 or s1!=s2: continue
            if b2[0]<=b1[0] and b1[1]<=b2[1] and (b1[1]-b1[0])<(b2[1]-b2[0]):
                drop.add(i)
    probes=[pr for i,pr in enumerate(keep) if i not in drop]
    PER={}
    for _,key,stag,blk in probes:
        PER.setdefault(key,[]).append(pca_block(key,stag,blk))
    hs=[]
    LAST_PROJ_RANK.clear()
    for key,vs in PER.items():
        P=orth(torch.cat(vs).T)
        LAST_PROJ_RANK[key]=int(P.shape[1])
        if key[0]=='a':
            def fh(mo,i_,o_,P=P):
                y,v1=o_
                yf=y.float().reshape(-1,D)
                return ((yf-(yf@P)@P.T).view(y.shape).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,P=P):
                yf=o_.float().reshape(-1,D)
                return (yf-(yf@P)@P.T).view(o_.shape).to(o_.dtype)
        hs.append(MODS[key].register_forward_hook(fh))
    return hs

_MUS=None
@torch.no_grad()
def comp_means():
    global _MUS
    if _MUS is not None: return _MUS
    R=rows(); sums={}; hs=[]
    for key,mod in MODS.items():
        sums[key]=torch.zeros(D,device=DEV)
        def mk(key=key):
            def h(mo,i_,o_):
                y=o_[0] if isinstance(o_,tuple) else o_
                sums[key]+=y.detach().float().reshape(-1,D).sum(0)
            return h
        hs.append(mod.register_forward_hook(mk()))
    for i in range(0,R.shape[0],4):
        bb=R[i:i+4,:257].to(DEV)
        m(bb[:,:-1].contiguous(), bb[:,1:].contiguous())
    for h in hs: h.remove()
    _MUS={k:v/(R.shape[0]*256) for k,v in sums.items()}
    return _MUS

def mean_hooks(keys):
    mus=comp_means(); hs=[]
    for key in keys:
        mu=mus[key]
        if key[0]=='a':
            def fh(mo,i_,o_,mu=mu):
                y,v1=o_
                return (mu.expand_as(y).to(y.dtype),v1)
        else:
            def fh(mo,i_,o_,mu=mu):
                return mu.expand_as(o_).to(o_.dtype)
        hs.append(MODS[key].register_forward_hook(fh))
    return hs

@torch.no_grad()
def ce_sweep(hooks=(),tok=None):
    """Per-position CE. tok=None -> census grid (54272,)."""
    R=rows() if tok is None else tok
    ces=[]
    for i in range(0,R.shape[0],4):
        bb=R[i:i+4,:257].to(DEV)
        idx=bb[:,:-1].contiguous(); tg=bb[:,1:].reshape(-1)
        x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
        for blkm in m.transformer.h:
            x,v1=blkm(x,v1,x0)
        lg=(30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)).float()
        ces.append(F.cross_entropy(lg.view(-1,lg.size(-1)),tg,
                                   reduction='none').cpu())
    for h in hooks: h.remove()
    return torch.cat(ces).float()

def leaf_ablate(tag):
    """dCE vector for ablating the leaf's own machinery (its 4 probes)."""
    return ce_sweep(proj_hooks(leaf(tag)['top_probes']))-base_ce()

def member_scores(tag):
    lf=leaf(tag); sc=lf['score']
    msc=sc[sc.abs()>=sc.abs().quantile(0.85)]
    assert len(msc)==len(lf['member'])
    return msc

def sign_stats(tag,d):
    lf=leaf(tag); mem=lf['member']; msc=member_scores(tag)
    sl=lf['slice']
    mm=torch.zeros(nflat(),dtype=torch.bool); mm[mem]=True
    slm=torch.zeros(nflat(),dtype=torch.bool); slm[sl]=True
    npos=int((msc>0).sum()); nneg=int((msc<0).sum())
    am=float(d[mm].abs().mean()); ag=float(d[~slm].abs().mean())
    return {'abs_dce_members':round(am,3),
            'abs_dce_offslice':round(ag,3),
            'concentration':round(am/max(ag,1e-4),2),
            'dce_members':round(float(d[mm].mean()),3),
            'dce_pos':round(float(d[mem[msc>0]].mean()),3) if npos else None,
            'dce_neg':round(float(d[mem[msc<0]].mean()),3) if nneg else None,
            'n_pos':npos,'n_neg':nneg,
            'minority_share':round(min(npos,nneg)/max(len(mem),1),3),
            'base_ce_member_mean':round(float(base_ce()[mem].mean()),3),
            'base_ce_frac_lt3':round(float((base_ce()[mem]<3).float()
                                           .mean()),3)}

def examples_filtered(tag,d,kind,n=5,seed=11):
    """Mechanical class-filtered member draw (2026-08-20, from
    wave-2 reviewer friction: hand-built class draws varied by
    reviewer). kind: 'subword' (target has no leading space and is
    alphabetic), 'space_word', 'digit', 'punct', 'capitalized'
    (leading space + uppercase initial), 'newline'. Returns the
    same shape as examples()['rand']: seeded random draw over
    matching members, never outcome-ordered."""
    lf=leaf(tag); mem=lf['member']; R=rows()
    def match(t):
        s=d1(int(t)); st=s.strip()
        if kind=='subword': return (not s.startswith(' ')) and st.isalpha()
        if kind=='space_word': return s.startswith(' ') and st.isalpha()
        if kind=='digit': return st.isdigit()
        if kind=='punct': return bool(st) and not any(
            c.isalnum() for c in st)
        if kind=='capitalized': return s.startswith(' ') and \
            bool(st) and st[:1].isupper()
        if kind=='newline': return chr(10) in s
        raise ValueError(f'unknown kind {kind}')
    hits=[]
    for gi in mem.tolist():
        r_,p_=gi//256,gi%256
        if match(R[r_,p_+1]): hits.append(gi)
    if not hits: return {'kind':kind,'n_available':0,'draw':[]}
    g=torch.Generator().manual_seed(seed)
    pick=[hits[i] for i in torch.randperm(len(hits),
          generator=g)[:n].tolist()]
    out=[]
    for gi in pick:
        pre,tgt,_=context(gi)
        out.append({'gi':gi,'context':pre[-70:],'target':tgt,
                    'dce':round(float(d[gi]),2)})
    return {'kind':kind,'n_available':len(hits),'draw':out}

def story_test_class(tag,d,kind,pred_help,seeds=(1,2,3,4,11),
                     n=5,n_tests=1):
    """Robust behavioral test (2026-08-20, from wave-3 friction:
    a single seed-11 n=5 draw passes by chance ~2/5 of the time).
    Runs story_test on a filtered draw for each seed AND on the
    FULL population of that class. ROBUST requires both: >=60% of
    seeds pass and the whole-population binomial p <= 0.10."""
    from math import comb
    per=[]
    for s in seeds:
        f=examples_filtered(tag,d,kind,n=n,seed=s)
        if not f['draw']: continue
        st=story_test(tag,d,[x['gi'] for x in f['draw']],
                      [pred_help]*len(f['draw']))
        per.append({'seed':s,'hits':st['hits'],'n':st['n'],
                    'p_value':st['p_value']})
    frac=(sum(p['p_value']<=0.10 for p in per)/len(per)
          if per else 0.0)
    fall=examples_filtered(tag,d,kind,n=10**9,seed=0)
    gis=[x['gi'] for x in fall['draw']]
    pop=story_test(tag,d,gis,[pred_help]*len(gis)) if gis else \
        {'n':0,'hits':0,'p_value':1.0}
    # ROBUST (v1, deprecated 2026-08-20): the seed-stability leg is
    # underpowered by construction -- a true 84% effect draws 5/5
    # only ~41% of the time, so it demoted a real punctuation push
    # whose whole-population p was ~0 (wave-3 reviewer catch).
    # ROBUST_V2 gates on the population test, which uses EVERY
    # member of the class (no draw noise), plus a minimum class
    # size so n=3 populations cannot pass. Seed sweep is kept as a
    # reported diagnostic.
    return {'kind':kind,'pred_help':bool(pred_help),
            'n_available':fall['n_available'],'per_seed':per,
            'seed_pass_frac':round(frac,2),'population':pop,
            'ROBUST':bool(frac>=0.6 and pop['p_value']<=0.10),
            'n_tests':n_tests,
            'alpha':round(0.10/max(n_tests,1),4),
            'margin':round(pop['hits']/max(pop['n'],1)
                           -pop['base_rate_help'],3),
            'ROBUST_V2':bool(pop['p_value']<=0.10/max(n_tests,1)
                             and fall['n_available']>=10),
            'gate_note':'use ROBUST_V2; ROBUST v1 is underpowered'}

def story_test(tag,d,gis,pred_help):
    """Base-rate significance for a behavioral story (2026-08-20,
    from the wave-2 base-rate objection: two-signed leaves give
    ~50% hits by chance). gis: member indices scored; pred_help:
    list of bools (story predicts CE DOWN at that position).
    Returns hits, base-rate expectation, and a binomial tail
    p-value (Poisson-binomial approximated at the mean rate)."""
    from math import comb
    mem=leaf(tag)['member']
    p_help=float((d[mem]<0).float().mean())
    hits=0; exp=0.0
    for gi,ph in zip(gis,pred_help):
        act=float(d[int(gi)])<0
        hits+=int(act==bool(ph))
        exp+=p_help if ph else (1-p_help)
    n=len(gis); pbar=exp/max(n,1)
    pval=sum(comb(n,k)*pbar**k*(1-pbar)**(n-k)
             for k in range(hits,n+1)) if n else 1.0
    return {'n':n,'hits':hits,'base_rate_help':round(p_help,3),
            'expected_hits':round(exp,2),
            'p_value':round(float(pval),4),
            'beats_base_rate':bool(pval<=0.10)}

def sign_stats_half(tag,d,lo,hi):
    """v2 identity gate helper: sign_stats restricted to member/
    off-slice positions whose row is in [lo,hi). Additive
    (2026-08-20, per wave-2 friction)."""
    lf=leaf(tag); mem=lf['member']; sl=lf['slice']
    NF=nflat()
    rowof=lambda idx: idx//256
    mm=torch.zeros(NF,dtype=torch.bool); mm[mem]=True
    slm=torch.zeros(NF,dtype=torch.bool); slm[sl]=True
    rows_=torch.arange(NF)//256
    band=(rows_>=lo)&(rows_<hi)
    am=float(d[mm&band].abs().mean())
    ag=float(d[(~slm)&band].abs().mean())
    return {'concentration':round(am/max(ag,1e-4),2),
            'abs_dce_members':round(am,3),
            'n_members':int((mm&band).sum())}

def examples(tag,d=None,ntop=3,nrand=3,seed=0):
    lf=leaf(tag); mem=lf['member']; msc=member_scores(tag)
    order=msc.abs().argsort(descending=True)
    g=torch.Generator().manual_seed(seed)
    rest=mem[order[ntop:]]
    rnd=rest[torch.randperm(len(rest),generator=g)[:nrand]]
    def ex(gi):
        gi=int(gi); pre,tgt,_=context(gi)
        e={'gi':gi,'context':pre[-70:],'target':tgt,
           'base_ce':round(float(base_ce()[gi]),2)}
        if d is not None: e['dce']=round(float(d[gi]),2)
        return e
    return {'top':[ex(g_) for g_ in mem[order[:ntop]]],
            'random':[ex(g_) for g_ in rnd],
            'rule':f'top-{ntop} by |score| + {nrand} seed-{seed} random'}

# ---------- feature library + rule search (ported from cl2) ----------
_FEAT=None
def surface_features():
    """Base predicate library on the census grid, flat bools."""
    global _FEAT
    if _FEAT is not None: return _FEAT
    R=rows(); tok2d=R[:,:256]; ntokr,Tr=tok2d.shape
    flat=tok2d.reshape(-1)
    uniq=flat.unique()
    def strfeat(fn):
        vals={int(u):fn(d1(int(u))) for u in uniq.tolist()}
        return torch.tensor([vals[int(t)] for t in flat.tolist()]) \
            .view(ntokr,Tr)
    NL=strfeat(lambda s: chr(10) in s)
    DIG=strfeat(lambda s: s.strip().isdigit())
    PUN=strfeat(lambda s: bool(s.strip()) and
                not any(c.isalnum() for c in s.strip()))
    UPI=strfeat(lambda s: s.startswith(' ') and s.strip()[:1].isupper())
    MID=strfeat(lambda s: (not s.startswith(' ')) and s.strip().isalpha())
    SPC=strfeat(lambda s: s.startswith(' '))
    SEEN=torch.zeros(ntokr,Tr,dtype=torch.bool)
    DNL=torch.zeros(ntokr,Tr,dtype=torch.long)
    lut={int(u):(chr(10) in d1(int(u))) for u in uniq.tolist()}
    for r_ in range(ntokr):
        seen=set(); dn=99
        row=tok2d[r_].tolist()
        for t_ in range(Tr):
            SEEN[r_,t_]=row[t_] in seen; seen.add(row[t_])
            dn=0 if lut[row[t_]] else min(dn+1,99)
            DNL[r_,t_]=dn
    B={'is_digit':DIG,'is_punct':PUN,'upper_initial':UPI,
       'mid_word':MID,'starts_space':SPC,'is_newline':NL,
       'seen_before':SEEN,'prev_newline':torch.roll(NL,1,dims=1),
       'dist_nl_le2':DNL<=2,'dist_nl_ge6':DNL>=6}
    # 10 mechanical class labels (cls_rows port -- classify the TARGET)
    from circuit_dictionary import CLS as CLS9
    CM=torch.zeros(ntokr,Tr,dtype=torch.long)
    for r_ in range(ntokr):
        tt_=R[r_,:257].tolist()
        for pos in range(Tr):
            t=tt_[pos+1]; pch=tt_[pos]
            tg=d1(t); pv=d1(pch); st=tg.strip()
            if st.isdigit() and not tg.startswith(' '): k=0
            elif st in (')',']') and any(bch in ''.join(
                d1(x) for x in tt_[max(0,pos-60):pos+1])
                for bch in ('(','[')): k=1
            elif chr(10) in tg: k=2
            elif tg in ('.','!','?'): k=3
            elif tg==',': k=4
            elif (tg.startswith(' ') and st[:1].isupper() and
                  (pv.strip()[:1].isupper() if pv.strip() else False)): k=5
            elif t==pch: k=6
            elif (not tg.startswith(' ')) and st.isalpha(): k=7
            elif t in tt_[:pos+1]: k=8
            else: k=9
            CM[r_,pos]=k
    for k in range(10): B[f'class_{CLS9[k]}']=(CM==k)
    L0={k:v.reshape(-1) for k,v in B.items()}
    # registry features (features.json): named, versioned, append-only
    try:
        reg=json.load(open(PT+'features.json'))
        env={'torch':torch,'F':F,'flat':flat,'tok2d':tok2d,
             'roll':torch.roll,'L0':dict(L0),'d1':d1}
        for nm,spec in reg.get('features',{}).items():
            if spec.get('kind')=='expr':
                try: L0[nm]=eval(spec['expr'],env).reshape(-1)
                except Exception as e: print(f'feature {nm} failed: {e}')
    except FileNotFoundError: pass
    L=dict(L0)
    for nm,v in L0.items():
        v2=v.view(ntokr,Tr)
        L['prev1_'+nm]=torch.roll(v2,1,dims=1).reshape(-1)
        L['prev2_'+nm]=torch.roll(v2,2,dims=1).reshape(-1)
    # second pass: program-kind registry features (composed of L names)
    try:
        reg=json.load(open(PT+'features.json'))
        for nm,spec in reg.get('features',{}).items():
            if spec.get('kind')=='program' and nm not in L:
                L[nm]=run_program(L,spec['program'])
    except FileNotFoundError: pass
    _FEAT=L
    return L

def run_program(f,prog,nflat=None):
    if nflat is None: nflat=next(iter(f.values())).numel()
    mm=torch.zeros(nflat,dtype=torch.bool)
    for conj in prog:
        cm=torch.ones(nflat,dtype=torch.bool)
        for pred in conj:
            neg=pred.startswith('NOT ')
            nm=pred[4:] if neg else pred
            if nm not in f: cm&=False; continue
            cm&=(~f[nm] if neg else f[nm])
        mm|=cm
    return mm

def register_feature(name,spec):
    p=PT+'features.json'
    with _lock('features'):
        reg=json.load(open(p)) if os.path.exists(p) else {'features':{}}
        if name in reg['features']:
            raise ValueError(f'feature {name} already registered')
        reg['features'][name]=spec
        json.dump(reg,open(p,'w'),indent=1)

def rule_search(f,pos,neg,nflat=None):
    if nflat is None: nflat=next(iter(f.values())).numel()
    names_=list(f)
    def acc(mask):
        tp=float(mask[pos].float().mean())
        fp=float(mask[neg].float().mean())
        return (tp+(1-fp))/2
    def best_conj(exclude):
        cur=torch.ones(nflat,dtype=torch.bool); used=[]
        for _ in range(3):
            bn,bb,ba=None,None,0
            for nm2 in names_:
                if nm2 in used or nm2 in exclude: continue
                for pol in (True,False):
                    a2=acc(cur&(f[nm2]==pol))
                    if a2>ba: ba,bn,bb=a2,nm2,pol
            if bn is None: break
            nm3=bn if bb else 'NOT '+bn
            if used and ba<=acc(cur)+0.01: break
            cur=cur&(f[bn]==bb); used.append(nm3)
        return cur,used,acc(cur)
    c1,u1,a1=best_conj([])
    c2,u2,a2=best_conj([u.replace('NOT ','') for u in u1])
    both=c1|c2
    if acc(both)>a1+0.02 and len(u2)>0: return both,[u1,u2]
    return c1,[u1]

def leaf_program(tag,f=None,seed=3):
    """Doc-disjoint program search for a leaf. Returns dict with
    held-out balanced acc, shuffled-label null, and program."""
    if f is None: f=surface_features()
    lf=leaf(tag); mem=lf['member']; bv=base_ce()
    ntokr=rows().shape[0]
    memflat=torch.zeros(globals()['nflat'](),dtype=torch.bool); memflat[mem]=True
    g9=torch.Generator().manual_seed(seed)
    lo,hi=bv[mem].quantile(0.1),bv[mem].quantile(0.9)
    nonidx=torch.nonzero((~memflat)&(bv>=lo)&(bv<=hi)).squeeze(1)
    nonidx=nonidx[torch.randperm(len(nonidx),generator=g9)[:len(mem)]]
    # doc-disjoint half: docid parity on the diverse tree (rows of
    # one document are adjacent there; row parity leaks). Falls back
    # to row parity on the old 212-row state (its rows are one doc).
    if ntokr==1000 and os.path.exists(PT+'curated_rows.pt'):
        docid=torch.load(PT+'curated_rows.pt',map_location='cpu',
                         weights_only=False)['docid']
        halfrow=(docid%2==0)
    else:
        halfrow=(torch.arange(ntokr)%2==0)
    half=halfrow[:,None].expand(-1,256).reshape(-1)
    posA=mem[half[mem]]; posB=mem[~half[mem]]
    negA=nonidx[half[nonidx]]; negB=nonidx[~half[nonidx]]
    if min(len(posA),len(posB),len(negA),len(negB))<30:
        return {'tag':tag,'ok':False,'reason':'too few per split'}
    mask,prog=rule_search(f,posA,negA)
    bacc=(float(mask[posB].float().mean())
          +1-float(mask[negB].float().mean()))/2
    lab=torch.cat([posA,negA])
    lab=lab[torch.randperm(len(lab),generator=g9)]
    mn,_=rule_search(f,lab[:len(posA)],lab[len(posA):])
    null=(float(mn[posB].float().mean())
          +1-float(mn[negB].float().mean()))/2
    return {'tag':tag,'ok':True,'bacc':round(bacc,3),
            'null':round(null,3),'program':prog}

# ---------- circuit registry ----------
import fcntl
from contextlib import contextmanager
from circuit_registry_v2 import (
    append_artifacts,
    append_claim_revision,
    append_evidence_event,
    rebuild_registry_v2,
    validate_v2 as validate_circuit_v2,
    write_behavior_circuit,
)
@contextmanager
def _lock(name):
    os.makedirs(CIRC,exist_ok=True)
    fh=open(CIRC+'.'+name+'.lock','w')
    fcntl.flock(fh,fcntl.LOCK_EX)
    try: yield
    finally: fcntl.flock(fh,fcntl.LOCK_UN); fh.close()

def circuit_path(tag): return CIRC+tag.replace('.','_')+'.json'

def rebuild_registry():
    """Idempotent generated view over v1 census and v2 behavior records."""
    return rebuild_registry_v2()
def write_circuit(tag,updates):
    """Merge updates into circuits/<tag>.json + registry row.
    Concurrency-safe: per-tag work is fine in parallel; the registry
    is rebuilt from files under a lock (no read-modify-write race)."""
    os.makedirs(CIRC,exist_ok=True)
    p=circuit_path(tag)
    nr=rows().shape[0]
    inst='212row-v1' if nr==212 else f'diverse-{nr}row-v1'
    doc=json.load(open(p)) if os.path.exists(p) else {
        'schema_version':1,'tag':tag,
        'tree':{'instance':inst,'n_rows':nr}}
    for k,v in updates.items():
        if k=='certification':
            # append-only: concatenate, dedup by (test, source, date)
            old=doc.get('certification',[])
            keys={(e.get('test'),e.get('source'),e.get('date'))
                  for e in old}
            doc[k]=old+[e for e in v if (e.get('test'),e.get('source'),
                                         e.get('date')) not in keys]
        elif isinstance(v,dict) and isinstance(doc.get(k),dict):
            doc[k]={**doc[k],**v}
        else:
            doc[k]=v
    if 'members' not in doc and tag in state()['by_tag']:
        doc['members']={'n':leaf(tag)['n_members']}
    json.dump(doc,open(p,'w'),indent=1)
    rebuild_registry()
    return p

# ---------- canonical copies (do NOT re-implement in scripts) ----------
# Everything below existed as near-identical copies in 5+ experiment
# scripts before the 2026-08-19 cleanup. New scripts must import these.

@torch.no_grad()
def head_parts(li,X,v1):
    """Recompute one attention layer's per-head patterns and mixed
    values. Returns (z, vm): z (B,9,T,128) head outputs pre-c_proj,
    vm (B,T,9,128) lambda-mixed values. THE canonical recompute --
    verified exact vs the real forward (353: delta +0.0000)."""
    import sys as _s
    are=_s.modules[type(m.transformer.h[0].attn).__module__] \
        .apply_rotary_emb
    at=m.transformer.h[li].attn
    Bb,Tq=X.shape[0],X.shape[1]
    v=at.c_v(X).view(Bb,Tq,9,128)
    vm=(1-at.lamb)*v+at.lamb*(v1.view_as(v) if v1 is not None else v)
    cos,sin=at.rotary(at.c_q(X).view(Bb,Tq,9,128))
    qf=F.rms_norm(at.c_q(X).view(Bb,Tq,9,128),(128,))
    kf=F.rms_norm(at.c_k(X).view(Bb,Tq,9,128),(128,))
    qf,kf=are(qf,cos,sin),are(kf,cos,sin)
    q2=F.rms_norm(at.c_q2(X).view(Bb,Tq,9,128),(128,))
    k2=F.rms_norm(at.c_k2(X).view(Bb,Tq,9,128),(128,))
    q2,k2=are(q2,cos,sin),are(k2,cos,sin)
    sc=torch.einsum('bqhd,bkhd->bhqk',qf.float(),kf.float())/128
    s2=torch.einsum('bqhd,bkhd->bhqk',q2.float(),k2.float())/128
    pat=(sc*s2)*torch.tril(torch.ones(Tq,Tq,device=X.device))
    z=torch.einsum('bhqk,bkhd->bhqd',pat,vm.float())
    return z,vm.float()

def mean_head_hooks(li,hds):
    """Within-batch MEAN ablation of specific heads (content killed,
    magnitude kept). LESSON 353/354: never zero-ablate heads -- zeroing
    is a magnitude shock that reads uniform across heads."""
    at=m.transformer.h[li].attn
    def fh(mo_,args,out,at=at,hds=hds):
        y,v1r=out
        X=args[0]; v1=args[1] if args[1] is not None else v1r
        z,vm=head_parts(li,X,v1)
        for hd in hds: z[:,hd]=z[:,hd].mean(1,keepdim=True)
        Bb,Tq=X.shape[0],X.shape[1]
        yn=at.c_proj(z.transpose(1,2).contiguous()
                     .view(Bb,Tq,-1).to(X.dtype))
        return (yn,v1r)
    return [at.register_forward_hook(fh)]

def fresh_rows(n=120,start=5000):
    """n never-seen 513-token pile rows, dedup'd against FW (the
    canonical fresh-data recipe, Ledger 22)."""
    import tiktoken
    from datasets import load_dataset
    enc2=tiktoken.get_encoding('gpt2')
    ds=load_dataset('NeelNanda/pile-10k',split='train')
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    out=[]
    for di in range(start,10000):
        tk=enc2.encode_ordinary(ds[di]['text'])
        for s0 in range(0,len(tk)-513,513):
            row=tk[s0:s0+513]
            if tuple(row[:32]) in seen: continue
            out.append(row)
            if len(out)>=n: break
        if len(out)>=n: break
    return torch.tensor(out,dtype=torch.long)

def fineweb_rows(n=120,skip=0):
    """n never-seen 513-token FineWeb rows via streaming (the model's
    TRAINING distribution -- confirmed by the user 2026-08-19).
    Dedup'd against FW. Use for IN-DISTRIBUTION fresh legs;
    fresh_rows() (pile) remains as the harder mildly-OOD leg.
    Always say which one a fresh number used."""
    from datasets import load_dataset
    e=enc()
    ds=load_dataset('HuggingFaceFW/fineweb',split='train',
                    streaming=True)
    seen={tuple(FW[r,:32].tolist()) for r in range(FW.shape[0])}
    out=[]; sk=0
    for ex in ds:
        if sk<skip: sk+=1; continue
        tk=e.encode_ordinary(ex['text'])
        for s0 in range(0,len(tk)-513,513):
            row=tk[s0:s0+513]
            if tuple(row[:32]) in seen: continue
            out.append(row)
            if len(out)>=n: break
        if len(out)>=n: break
    return torch.tensor(out,dtype=torch.long)

def ioi_prompts():
    """The canonical 96-prompt IOI set (8 pairs x 2 orders x 6
    templates). Returns [(text, io_token_id, s_token_id)]."""
    import itertools
    e=enc()
    names=[' Mary',' John',' Anna',' Peter',' Sarah',' Tom',
           ' Alice',' Bob']
    T9=['When{A} and{B} went to the store,{B} gave the drink to',
        'When{A} and{B} got home,{B} handed the keys to',
        'After{A} and{B} left the party,{B} gave the coat to',
        'Then{A} and{B} went to the park, and{B} threw the ball to',
        'While{A} and{B} were cooking,{B} passed the salt to',
        'When{A} and{B} finished lunch,{B} gave the bill to']
    out=[]
    for A,B in list(itertools.combinations(names,2))[:8]:
        for a,b in ((A,B),(B,A)):
            for tpl in T9:
                out.append((tpl.replace('{A}',a).replace('{B}',b),
                            e.encode(a)[0],e.encode(b)[0]))
    return out


def score_bar(name,value,bar,denom=None,n=None,min_n=10,
              min_denom=None,ref=None):
    """Score a registered prediction with the two guards this
    program keeps tripping over (writeups 465, 500).

    UNEVALUABLE, not FAILED, when:
      * the comparison class is empty or thin (n < min_n) -- a bar
        cannot be scored on a class the sample never populated;
      * the bar is a ratio and its denominator is near zero
        (|denom| < min_denom, default 10% of |ref| if given, else
        0.05 x |value|) -- a quotient of two numbers straddling
        zero carries no information regardless of how large it is.
    Returns (verdict, note) with verdict in HELD/FAILED/UNEVALUABLE
    and prints the line, so a run cannot quietly score junk.
    """
    note=''
    if n is not None and n<min_n:
        v='UNEVALUABLE'; note=f'class has n={n} < {min_n}'
    elif denom is not None:
        md=(min_denom if min_denom is not None else
            (0.1*abs(ref) if ref else 0.05*abs(value)))
        if abs(denom)<md:
            v='UNEVALUABLE'
            note=(f'denominator {denom:.5f} is below {md:.5f} -- '
                  f'report the pair, not the quotient')
        else:
            v='HELD' if value>=bar else 'FAILED'
    else:
        v='HELD' if value>=bar else 'FAILED'
    print(f'({name}) {value} vs bar {bar}: {v}'
          +(f' [{note}]' if note else ''),flush=True)
    return v,note


def writer_coeffs(li,kind='a'):
    """EXACT coefficient of each writer's output in the residual
    entering block li's attention (kind='a') or MLP (kind='m').

    Added 2026-08-20 (writeup 503). Every earlier decomposition in
    this program used lam0 of the CURRENT block for every writer,
    which is wrong: each block rescales the running residual as
    x = lam0*x + lam1*x0, so writer j's output arrives multiplied
    by the PRODUCT of lam0 over blocks j+1..li. With lam0=0.0127 at
    block 1 the error is four orders of magnitude for layer-0
    writers. The flat version reconstructs the layer-12 attention
    input to 68% relative error; this one to 1.2e-7.
    """
    H=m.transformer.h; L=len(H)
    lam=[H[j].lambdas.detach().float() for j in range(L)]
    def prod(a,b):
        p=1.0
        for k in range(a,b+1): p*=float(lam[k][0])
        return p
    c={}
    for j in range(li):
        cj=prod(j+1,li)
        c[f'a{j}']=cj; c[f'm{j}']=cj
    wte=prod(0,li)
    for j in range(0,li+1):
        wte+=prod(j+1,li)*float(lam[j][1])
    c['wte']=wte
    if kind=='m':
        c[f'a{li}']=1.0      # written after the block's lambda mix
    return c


def writer_parts(li,E,outs,kind='a'):
    """Exact additive parts of block li's component input.
    E is rms_norm(wte(idx)) as float; outs maps 'a3'/'m3' to that
    component's captured output. Returns {writer: tensor}."""
    c=writer_coeffs(li,kind)
    parts={}
    for w,cf in c.items():
        src=E if w=='wte' else outs.get(w)
        if src is None: continue
        parts[w]=cf*src
    return parts


def check_parts(parts,X,tol=1e-4,label=''):
    """Verify a writer decomposition reproduces the real input.
    Returns (ok, relerr) and PRINTS -- a decomposition that is
    merely close is the failure mode of writeups 443/447/503."""
    import torch.nn.functional as _F
    tot=sum(parts.values())
    Xr=_F.rms_norm(tot,(tot.shape[-1],))
    rel=float((Xr-X.float()).norm()/X.float().norm().clamp_min(1e-9))
    ok=rel<=tol
    print(f'  [check_parts{" "+label if label else ""}] relative '
          f'error {rel:.3e} -> {"ok" if ok else "FAILED, run is VOID"}',
          flush=True)
    return ok,rel


def assert_disjoint(fit_rows,price_rows,prefix=64,label=''):
    """Verify that no priced row appears in the fitting corpus.

    Added 2026-08-20 (writeup 548) after the SECOND occurrence of
    the same error in this program: tables fitted on rows() were
    priced on fineweb_rows(48), and 33 of those 48 rows were
    verbatim in the fitting set, because both draw from the same
    FineWeb slice and the loader names merely LOOK different.
    Compares the first `prefix` tokens of each row, which is
    sufficient to identify a row and cheap. Returns (ok, n_shared)
    and PRINTS, so a run cannot quietly report contaminated costs.
    """
    fs={tuple(r[:prefix].tolist()) for r in fit_rows}
    shared=[i for i,r in enumerate(price_rows)
            if tuple(r[:prefix].tolist()) in fs]
    ok=not shared
    print(f'  [assert_disjoint{" "+label if label else ""}] '
          f'{len(shared)} of {len(price_rows)} priced rows appear '
          f'in the fitting corpus -> '
          f'{"ok" if ok else "CONTAMINATED, run is VOID"}',
          flush=True)
    return ok,len(shared)
