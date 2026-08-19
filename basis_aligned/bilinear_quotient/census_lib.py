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
def state():
    global _st
    if _st is None:
        _st=torch.load(PT+'census_state.pt',map_location='cpu')
        _st['by_tag']={lf['tag']:lf for lf in _st['leaves']}
    return _st

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
        else torch.arange(54272)
    Y=capture_out(key)[sl].float().to(DEV)
    _,_,Vh=torch.linalg.svd((Y-Y.mean(0))[:20000],full_matrices=False)
    _PCA[kk]=Vh[blk[0]:blk[1]]
    return _PCA[kk]

def proj_hooks(probes):
    """probes: list of ('pca',key,stag,(s0,s1)) (strings also ok)."""
    probes=[eval(p) if isinstance(p,str) else p for p in probes]
    PER={}
    for _,key,stag,blk in probes:
        PER.setdefault(key,[]).append(pca_block(key,stag,blk))
    hs=[]
    for key,vs in PER.items():
        P=orth(torch.cat(vs).T)
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
    mm=torch.zeros(54272,dtype=torch.bool); mm[mem]=True
    slm=torch.zeros(54272,dtype=torch.bool); slm[sl]=True
    npos=int((msc>0).sum()); nneg=int((msc<0).sum())
    return {'abs_dce_members':round(float(d[mm].abs().mean()),3),
            'abs_dce_offslice':round(float(d[~slm].abs().mean()),3),
            'dce_members':round(float(d[mm].mean()),3),
            'dce_pos':round(float(d[mem[msc>0]].mean()),3) if npos else None,
            'dce_neg':round(float(d[mem[msc<0]].mean()),3) if nneg else None,
            'n_pos':npos,'n_neg':nneg,
            'minority_share':round(min(npos,nneg)/max(len(mem),1),3),
            'base_ce_member_mean':round(float(base_ce()[mem].mean()),3),
            'base_ce_frac_lt3':round(float((base_ce()[mem]<3).float()
                                           .mean()),3)}

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
    _FEAT=L
    return L

def rule_search(f,pos,neg,nflat=54272):
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
    memflat=torch.zeros(54272,dtype=torch.bool); memflat[mem]=True
    g9=torch.Generator().manual_seed(seed)
    lo,hi=bv[mem].quantile(0.1),bv[mem].quantile(0.9)
    nonidx=torch.nonzero((~memflat)&(bv>=lo)&(bv<=hi)).squeeze(1)
    nonidx=nonidx[torch.randperm(len(nonidx),generator=g9)[:len(mem)]]
    half=(torch.arange(ntokr)%2==0)[:,None].expand(-1,256).reshape(-1)
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
def circuit_path(tag): return CIRC+tag.replace('.','_')+'.json'
def write_circuit(tag,updates):
    """Merge updates into circuits/<tag>.json + registry row."""
    os.makedirs(CIRC,exist_ok=True)
    p=circuit_path(tag)
    doc=json.load(open(p)) if os.path.exists(p) else {
        'schema_version':1,'tag':tag,
        'tree':{'instance':'212row-v1','n_rows':212}}
    doc.update(updates)
    json.dump(doc,open(p,'w'),indent=1)
    rp=CIRC+'registry.json'
    reg=json.load(open(rp)) if os.path.exists(rp) else {'circuits':{}}
    lf=leaf(tag) if tag in state()['by_tag'] else {}
    reg['circuits'][tag]={'file':os.path.basename(p),
        'n_members':lf.get('n_members'),
        'headline':doc.get('story',{}).get('blind_name',''),
        'cert':len(doc.get('certification',[]))}
    json.dump(reg,open(rp,'w'),indent=1)
    return p
