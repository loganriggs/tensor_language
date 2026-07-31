"""NON-UNIFORM RANK ALLOCATION (RESULTS §87 follow-up, the fix §87 itself pointed at).
§87 measured the whole-model compression frontier with UNIFORM per-layer restricted cores
(128x/+1.46, 16x/+0.80, 4x/+0.35), but §73 shows rank need is wildly non-uniform (late layers
low-rank, early hub layers high-rank). QUESTION: at a FIXED total core-parameter budget, how much
does allocating (Kin_l, Kout_l) per layer by need beat uniform allocation?

Machinery VERBATIM from qk_allcore_restrict.py: per-layer input/output train-gram SVD bases,
per-position held means, all-layer simultaneous restriction forward, core accounting
core = sum_l Kout_l * Kin_l*(Kin_l+1)/2.  Only generalization: PIN/POUT lists may carry
per-layer different widths (the forward already indexes per layer) and POUT entries may be None
per layer.  TRAIN FW[0:256] grams, held-back FW[448:600] eval, paired SE, batch<=6.

Rules at each budget B (B matched to §87's 128x/16x/4x core sizes):
  (a) UNIFORM  — the §87 baseline, re-run to verify +1.456 at (288,144);
  (b) SPECTRAL — per-layer (Kin_l,Kout_l) = smallest ranks capturing fraction f of that layer's
      input/output gram trace; f binary-searched so total core params <= B;
  (c) CAUSAL   — like (b) but the per-layer retained-fraction target is boosted by the layer's
      causal floor (qk_allterm_census.json floor_dCE): f_l = 1-(1-f)^{w_l}, w_l = floor_l/mean.
Caches bases/eigenvalues/means/base-CE to qk_rank_alloc_cache.pt for qk_rank_alloc_2.py."""
import json, os, subprocess, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/qk_mdl')
from tier2_model import load_elriggs, rope_tables, apply_rot

def gpu_guard():
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits']).decode().split('\n')[0].strip())
        if free>=4500: return free
        print(f"GPU guard: only {free} MiB free — sleeping 20s",flush=True); time.sleep(20)
print("GPU guard: free",gpu_guard(),"MiB",flush=True)

torch.manual_seed(0); DEV='cuda'; QK='/workspace/tensor_language/basis_aligned/qk_mdl'
m,cfg=load_elriggs('bilin18'); NH,HD,D=cfg['n_head'],cfg['n_embd']//cfg['n_head'],cfg['n_embd']
V=cfg['vocab_size']; NL=len(m.transformer.h)
FW=torch.from_numpy(np.load('/workspace/tensor_language/data_fineweb_tokens.npy').astype(np.int64))
TRAIN=FW[0:256,:128].to(DEV); HELD=FW[448:600,:128].to(DEV); B0=6
GIN=[torch.zeros(D,D,device=DEV) for _ in range(NL)]; GOUT=[torch.zeros(D,D,device=DEV) for _ in range(NL)]
@torch.no_grad()
def fwd(idx, mode=None, PIN=None, POUT=None, MX=None, MO=None, collect=False, want_gram=False):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
    xs=[] if collect else None; ms=[] if collect else None
    for li in range(NL):
        blk=m.transformer.h[li]; x=blk.lambdas[0]*x+blk.lambdas[1]*x0; a=blk.attn; hcur=F.rms_norm(x,(D,))
        def qk(l): z=F.rms_norm(l(hcur).view(B,T,NH,HD),(HD,)); return apply_rot(z,cosb,sinb)
        v=a.c_v(hcur).view(B,T,NH,HD)
        if v1 is None: v1=v
        v=(1-a.lamb)*v+a.lamb*v1.view_as(v)
        q,k,q2,k2=qk(a.c_q),qk(a.c_k),qk(a.c_q2),qk(a.c_k2)
        s1=torch.einsum('bqhd,bkhd->bhqk',q,k)/HD; s2=torch.einsum('bqhd,bkhd->bhqk',q2,k2)/HD
        pat=(s1*s2).masked_fill(~mask,0.0); yh=torch.einsum('bhqk,bkhd->bqhd',pat,v)
        x=x+a.c_proj(yh.reshape(B,T,-1))
        if want_gram: GIN[li].add_(torch.einsum('btd,bte->de',x,x))
        if collect: xs.append(x.clone())
        if mode is not None:
            xr=MX[li].unsqueeze(0)+((x-MX[li].unsqueeze(0))@PIN[li])@PIN[li].T
            mo=blk.mlp(F.rms_norm(xr,(D,)))
            if POUT is not None and POUT[li] is not None:
                mo=MO[li].unsqueeze(0)+((mo-MO[li].unsqueeze(0))@POUT[li])@POUT[li].T
        else:
            mo=blk.mlp(F.rms_norm(x,(D,)))
        if want_gram: GOUT[li].add_(torch.einsum('btd,bte->de',mo,mo))
        if collect: ms.append(mo.clone())
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce,xs,ms
print("gram pass ...",flush=True)
for i in range(0,TRAIN.shape[0],B0): fwd(TRAIN[i:i+B0],want_gram=True)
EVin=[];EVout=[];INb=[];OUTb=[]
for l in range(NL):
    w,vec=torch.linalg.eigh(GIN[l]); EVin.append(w.flip(0).clamp_min(0)); INb.append(vec.flip(1))
    w,vec=torch.linalg.eigh(GOUT[l]); EVout.append(w.flip(0).clamp_min(0)); OUTb.append(vec.flip(1))
print("mean pass ...",flush=True)
S,T=HELD.shape
MXs=[torch.zeros(T,D,device=DEV) for _ in range(NL)]; MOs=[torch.zeros(T,D,device=DEV) for _ in range(NL)]
for i in range(0,S,B0):
    _,xs,ms=fwd(HELD[i:i+B0],collect=True)
    for l in range(NL): MXs[l]+=xs[l].sum(0); MOs[l]+=ms[l].sum(0)
MX=[t/S for t in MXs]; MO=[t/S for t in MOs]
bce=[]
for i in range(0,S,B0): ce,_,_=fwd(HELD[i:i+B0]); bce.append(ce.cpu())
base=torch.cat(bce,0)
def dstat(ce):
    d=(ce-base).flatten().double(); return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
torch.save({'INb':[b.cpu() for b in INb],'OUTb':[b.cpu() for b in OUTb],
            'EVin':[e.cpu() for e in EVin],'EVout':[e.cpu() for e in EVout],
            'MX':[t.cpu() for t in MX],'MO':[t.cpu() for t in MO],'base':base},
           f'{QK}/qk_rank_alloc_cache.pt')
print("cache saved",flush=True)

# ---------------- allocation rules ----------------
cumin=[ (e/e.sum()).cumsum(0).cpu().numpy() for e in EVin ]
cumout=[ (e/e.sum()).cumsum(0).cpu().numpy() for e in EVout ]
floors=[json.load(open(f'{QK}/qk_allterm_census.json'))['layers'][str(l)]['floor_dCE'] for l in range(NL)]
Wc=[f/(sum(floors)/NL) for f in floors]
KMIN_IN,KMIN_OUT=4,4
def ranks_for(f, weights=None):
    Kins=[];Kouts=[]
    for l in range(NL):
        fl=f if weights is None else 1.0-(1.0-f)**weights[l]
        fl=min(max(fl,0.0),1.0-1e-12)
        Kin=int(np.searchsorted(cumin[l],fl)+1); Kout=int(np.searchsorted(cumout[l],fl)+1)
        Kins.append(min(max(Kin,KMIN_IN),D)); Kouts.append(min(max(Kout,KMIN_OUT),D))
    return Kins,Kouts
def core_params(Kins,Kouts):
    return sum((Kouts[l] if Kouts[l] is not None else D)*Kins[l]*(Kins[l]+1)//2 for l in range(NL))
FULL=NL*D*D*(D+1)//2
def fit_budget(B, weights=None):
    lo,hi=1e-8,1.0-1e-12; best=None
    for _ in range(80):
        mid=0.5*(lo+hi); Kins,Kouts=ranks_for(mid,weights)
        if core_params(Kins,Kouts)<=B: best=(mid,Kins,Kouts); lo=mid
        else: hi=mid
    if best is None: best=(lo,)+ranks_for(lo,weights)
    return best
def runprof(Kins,Kouts):
    PIN=[INb[l][:,:Kins[l]].contiguous() for l in range(NL)]
    POUT=[(OUTb[l][:,:Kouts[l]].contiguous() if Kouts[l] is not None else None) for l in range(NL)]
    ces=[]
    for i in range(0,S,B0):
        ce,_,_=fwd(HELD[i:i+B0],mode='r',PIN=PIN,POUT=POUT,MX=MX,MO=MO); ces.append(ce.cpu())
    return torch.cat(ces,0)

BUDGETS={'128x':NL*144*288*289//2,'16x':NL*288*576*577//2,'4x':NL*1152*576*577//2}
UNIFORM={'128x':(288,144),'16x':(576,288),'4x':(576,None)}
res={'base_CE':float(base.mean()),'floors':floors,'causal_weights':[round(w,4) for w in Wc],
     'spectra':{'in_top4_frac':[round(float(cumin[l][3]),4) for l in range(NL)],
                'out_top4_frac':[round(float(cumout[l][3]),4) for l in range(NL)]},
     'budgets':{k:int(v) for k,v in BUDGETS.items()},'full_params':int(FULL),'rules':{}}
for tag,B in BUDGETS.items():
    res['rules'][tag]={}
    # (a) uniform baseline (verify §87)
    Kin,Kout=UNIFORM[tag]
    Kins=[Kin]*NL; Kouts=[Kout]*NL
    mn,se=dstat(runprof(Kins,Kouts)); cp=core_params(Kins,Kouts)
    res['rules'][tag]['uniform']={'Kin':Kin,'Kout':Kout,'dCE':round(mn,4),'SE':round(se,5),
        'core_params':int(cp),'compression_x':round(FULL/cp,1)}
    print(f"[{tag}] UNIFORM ({Kin},{Kout}): dCE {mn:+.4f} ± {se:.5f} ({FULL/cp:.0f}x)",flush=True)
    # (b) spectral
    f,Kins,Kouts=fit_budget(B); mn,se=dstat(runprof(Kins,Kouts)); cp=core_params(Kins,Kouts)
    res['rules'][tag]['spectral']={'f':round(f,6),'Kin':Kins,'Kout':Kouts,'dCE':round(mn,4),
        'SE':round(se,5),'core_params':int(cp),'compression_x':round(FULL/cp,1)}
    print(f"[{tag}] SPECTRAL f={f:.4f}: dCE {mn:+.4f} ± {se:.5f} (params {cp/1e6:.0f}M, {FULL/cp:.0f}x)",flush=True)
    print("   Kin :",Kins,"\n   Kout:",Kouts,flush=True)
    # (c) causal-weighted
    f,Kins,Kouts=fit_budget(B,weights=Wc); mn,se=dstat(runprof(Kins,Kouts)); cp=core_params(Kins,Kouts)
    res['rules'][tag]['causal']={'f':round(f,6),'Kin':Kins,'Kout':Kouts,'dCE':round(mn,4),
        'SE':round(se,5),'core_params':int(cp),'compression_x':round(FULL/cp,1)}
    print(f"[{tag}] CAUSAL   f={f:.4f}: dCE {mn:+.4f} ± {se:.5f} (params {cp/1e6:.0f}M, {FULL/cp:.0f}x)",flush=True)
    print("   Kin :",Kins,"\n   Kout:",Kouts,flush=True)
    json.dump(res,open(f'{QK}/qk_rank_alloc.json','w'),indent=1)
json.dump(res,open(f'{QK}/qk_rank_alloc.json','w'),indent=1)
print("DONE",flush=True)
