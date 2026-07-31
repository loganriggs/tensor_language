"""NON-UNIFORM RANK ALLOCATION, part 2 — MEASURED-NEED (greedy marginal-cost) allocation.
Part 1 found both a-priori rules (gram-trace fraction, causal-floor weighting) LOSE to uniform:
the gram trace is dominated by large shared residual directions and is a bad proxy for functional
rank need (layer 1's input gram is so concentrated the spectral rule gave the hub Kin=4).
The honest 'allocate by need' rule uses MEASURED per-layer restriction cost curves: §87 showed
compounding is roughly additive, so single-layer costs should predict joint costs.
  1. For each layer alone (all others exact), measure dCE on a held SUBSAMPLE (36 seqs) at a
     ladder of (Kin,Kout) core sizes.
  2. Greedy allocator: per-layer lower convex hull of (params, cost); repeatedly buy the hull
     segment with the best cost-reduction per parameter until budget B is exhausted.
  3. Evaluate the resulting JOINT profile on the FULL held set at budgets 128x/32x/16x/8x/4x/2x;
     log-interpolate the budget needed for +0.35 and +0.15.
Machinery verbatim from qk_allcore_restrict.py via qk_rank_alloc.py (cache: bases, means, base CE);
only change: fwd skips restriction on layers whose PIN entry is None (per-layer exact)."""
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
HELD=FW[448:600,:128].to(DEV); B0=6
cache=torch.load(f'{QK}/qk_rank_alloc_cache.pt')
INb=[t.to(DEV) for t in cache['INb']]; OUTb=[t.to(DEV) for t in cache['OUTb']]
MX=[t.to(DEV) for t in cache['MX']]; MO=[t.to(DEV) for t in cache['MO']]
base=cache['base']
@torch.no_grad()
def fwd(idx, mode=None, PIN=None, POUT=None):
    B,T=idx.shape
    x=F.rms_norm(m.transformer.wte(idx),(D,)); x0=x; v1=None
    cos,sin=rope_tables(T,HD,DEV,x.dtype,'bf16'); cosb,sinb=cos[None,:,None,:],sin[None,:,None,:]
    mask=torch.tril(torch.ones(T,T,device=DEV,dtype=torch.bool))
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
        if mode is not None and PIN[li] is not None:
            xr=MX[li].unsqueeze(0)+((x-MX[li].unsqueeze(0))@PIN[li])@PIN[li].T
            mo=blk.mlp(F.rms_norm(xr,(D,)))
            if POUT is not None and POUT[li] is not None:
                mo=MO[li].unsqueeze(0)+((mo-MO[li].unsqueeze(0))@POUT[li])@POUT[li].T
        else:
            mo=blk.mlp(F.rms_norm(x,(D,)))
        x=x+mo
    logits=30*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30)
    ce=F.cross_entropy(logits[:,:-1].reshape(-1,V).float(),idx[:,1:].reshape(-1),reduction='none').view(B,T-1)
    return ce
S,T=HELD.shape; NSUB=36; SSUB=HELD[:NSUB]; base_sub=base[:NSUB]
def evalprof(Kins,Kouts,sub=False):
    """Kins[l] None => layer exact. Kouts[l] None => no output projection."""
    PIN=[(INb[l][:,:Kins[l]].contiguous() if Kins[l] is not None else None) for l in range(NL)]
    POUT=[(OUTb[l][:,:Kouts[l]].contiguous() if Kouts[l] is not None else None) for l in range(NL)]
    data=SSUB if sub else HELD; ref=base_sub if sub else base; ces=[]
    for i in range(0,data.shape[0],B0): ces.append(fwd(data[i:i+B0],mode='r',PIN=PIN,POUT=POUT).cpu())
    ce=torch.cat(ces,0); d=(ce-ref).flatten().double()
    return float(d.mean()),float(d.std()/np.sqrt(d.numel()))
def pstep(Kin,Kout): return (Kout if Kout is not None else D)*Kin*(Kin+1)//2
STEPS=[(36,18),(72,36),(144,72),(288,144),(432,216),(576,288),(576,576),(864,432),
       (864,864),(1152,576),(1152,1152)]
STEPS=sorted(STEPS,key=lambda s:pstep(*s))
FULL=NL*pstep(1152,1152)

# ---- 1. per-layer single-layer cost curves on the subsample ----
CURVES=[[None]*len(STEPS) for _ in range(NL)]
for l in range(NL):
    for si,(Kin,Kout) in enumerate(STEPS):
        if (Kin,Kout)==(1152,1152): CURVES[l][si]=0.0; continue
        Kins=[None]*NL; Kouts=[None]*NL; Kins[l]=Kin; Kouts[l]=Kout
        mn,_=evalprof(Kins,Kouts,sub=True); CURVES[l][si]=mn
    print(f"layer {l:2d} curve: "+" ".join(f"{c:+.3f}" for c in CURVES[l]),flush=True)

# ---- 2. greedy allocator on per-layer lower convex hulls ----
PAR=[pstep(*s) for s in STEPS]
def hull(l):
    """indices of lower convex hull of (params, cost), starting at step 0, ending at exact."""
    pts=[(PAR[si],CURVES[l][si],si) for si in range(len(STEPS))]
    h=[pts[0]]
    for p in pts[1:]:
        while len(h)>=2:
            (x1,y1,_),(x2,y2,_)=h[-2],h[-1]
            if (y2-y1)*(p[0]-x1)>=(p[1]-y1)*(x2-x1): h.pop()
            else: break
        if p[1]<h[-1][1] or p is pts[-1]: h.append(p)  # keep only cost-decreasing, always keep exact
    # enforce monotone decreasing cost along hull
    out=[h[0]]
    for p in h[1:]:
        if p[1]<out[-1][1]: out.append(p)
    return out  # list of (params, cost, step_index)
HULLS=[hull(l) for l in range(NL)]
def greedy(B):
    pos=[0]*NL  # index into HULLS[l]
    tot=sum(HULLS[l][0][0] for l in range(NL))
    while True:
        best=None
        for l in range(NL):
            if pos[l]+1<len(HULLS[l]):
                p0,c0,_=HULLS[l][pos[l]]; p1,c1,_=HULLS[l][pos[l]+1]
                if tot-p0+p1<=B:
                    slope=(c0-c1)/(p1-p0)
                    if best is None or slope>best[0]: best=(slope,l)
        if best is None: break
        l=best[1]; tot+=HULLS[l][pos[l]+1][0]-HULLS[l][pos[l]][0]; pos[l]+=1
    Kins=[];Kouts=[];pred=0.0
    for l in range(NL):
        si=HULLS[l][pos[l]][2]; Kin,Kout=STEPS[si]; pred+=CURVES[l][si]
        if (Kin,Kout)==(1152,1152): Kins.append(None);Kouts.append(None)
        else: Kins.append(Kin);Kouts.append(Kout)
    return Kins,Kouts,tot,pred

# ---- 3. joint evaluation at budgets ----
BUD={'128x':NL*144*288*289//2,'32x':FULL//32,'16x':NL*288*576*577//2,'8x':FULL//8,
     '4x':NL*1152*576*577//2,'2x':FULL//2}  # 128x/16x/4x match part 1's uniform budgets exactly
res=json.load(open(f'{QK}/qk_rank_alloc.json'))
res['greedy_curves']={'steps':STEPS,'params_per_step':PAR,
    'per_layer_cost_sub36':[[round(c,4) for c in CURVES[l]] for l in range(NL)]}
res['greedy']={}
for tag,B in BUD.items():
    Kins,Kouts,tot,pred=greedy(B)
    mn,se=evalprof(Kins,Kouts)
    res['greedy'][tag]={'Kin':[k if k is not None else 1152 for k in Kins],
        'Kout':[k if k is not None else 1152 for k in Kouts],
        'core_params':int(tot),'compression_x':round(FULL/tot,1),
        'predicted_additive_dCE':round(pred,4),'dCE':round(mn,4),'SE':round(se,5)}
    print(f"[{tag}] GREEDY: dCE {mn:+.4f} ± {se:.5f} (pred {pred:+.3f}, params {tot/1e6:.0f}M, {FULL/tot:.1f}x)",flush=True)
    print("   Kin :",res['greedy'][tag]['Kin'],"\n   Kout:",res['greedy'][tag]['Kout'],flush=True)
    json.dump(res,open(f'{QK}/qk_rank_alloc.json','w'),indent=1)

# ---- 4. thresholds: budget the best rule needs for +0.35 and +0.15 ----
pts=sorted([(res['greedy'][t]['core_params'],res['greedy'][t]['dCE']) for t in BUD],key=lambda x:x[0])
def cross(target):
    for (p0,c0),(p1,c1) in zip(pts,pts[1:]):
        if c0>=target>=c1:
            lp=np.log(p0)+(np.log(p1)-np.log(p0))*(c0-target)/(c0-c1)
            return float(np.exp(lp))
    return None
res['thresholds']={}
for tgt in [0.35,0.15]:
    p=cross(tgt)
    res['thresholds'][str(tgt)]={'core_params':None if p is None else int(p),
        'compression_x':None if p is None else round(FULL/p,2)}
    print(f"budget for dCE=+{tgt}: {'not bracketed' if p is None else f'{p/1e6:.0f}M params ({FULL/p:.1f}x compression)'}",flush=True)
json.dump(res,open(f'{QK}/qk_rank_alloc.json','w'),indent=1)
print("DONE",flush=True)
