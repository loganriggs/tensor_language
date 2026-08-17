"""HILLCLIMB ROUND 1 (user-directed): greedy per-layer rank allocation for the
tail replacement (layers 5-16; front L0-L4 and L17 stay real), sequential
refit, joint-scored. Baseline to beat: uniform refit r16 = +1.66 at 0.29M.

Method: fit each layer's rank-64 refit map front-to-back under the current
assignment (starting all-rank-4); greedy moves = raise one layer's rank along
(0,2,4,8,16,32,64) or lower it, chosen by approximate marginal dCE/dparam from
SVD truncations of the cached map, joint re-scored on acceptance; report exact
joint CE at budgets ~0.1M/0.3M/1M.

REGISTERED PREDICTIONS: (a) greedy at <=0.3M beats uniform-r16's +1.66 by
>= 0.10; (b) greedy dominates the uniform refit curve at all three budgets;
(c) the learned allocation is nonuniform: max/min rank ratio >= 4 (the layers
genuinely differ in needed capacity)."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import FW, DEV
import bilin18_pipe_refit as PR
D=1152
LAYERS=list(range(5,17))
RANKS=(0,2,4,8,16,32,64)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_hillclimb1_results.json')

def params_of(assign):
    return sum(2*D*r for r in assign.values())

@torch.no_grad()
def ce():
    tot,n=0.0,0
    for i in range(384,448,4):
        b=FW[i:i+4,:257].to(DEV)
        lg,_=PR.fwd_lin(b[:,:-1].contiguous())
        c=F.cross_entropy(lg.view(-1,lg.size(-1)), b[:,1:].reshape(-1))
        tot+=float(c)*(b.shape[1]-1)*b.shape[0]; n+=(b.shape[1]-1)*b.shape[0]
    return tot/n

def truncate(mp,r):
    if r==0:
        return {'W':torch.zeros(D,D,device=DEV),'bx':mp['bx'],'by':mp['by']}
    U,S,Vh=torch.linalg.svd(mp['W'])
    return {'W':U[:,:r]@torch.diag(S[:r])@Vh[:r],'bx':mp['bx'],'by':mp['by']}

@torch.no_grad()
def install(assign, maps):
    PR.LINS={li:truncate(maps[li],assign[li]) for li in LAYERS}

@torch.no_grad()
def refit_maps(assign):
    """Sequential front-to-back refit at rank 64 under the given assignment."""
    maps={}
    for li in LAYERS:
        PR.LINS={lj:truncate(maps[lj],assign[lj]) for lj in LAYERS if lj<li
                 and lj in maps}
        xs=[];ys=[]
        for i in range(0,48,6):
            _,cap=PR.fwd_lin(FW[i:i+6,:256].to(DEV), want=li)
            xs.append(cap[0]); ys.append(cap[1])
        X=torch.cat(xs); Y=torch.cat(ys)
        bx=X.mean(0); by=Y.mean(0)
        Xc=X-bx; Yc=Y-by
        lam=1e-2*float((Xc**2).mean())*Xc.shape[1]/Xc.shape[0]
        W=torch.linalg.solve(Xc.T@Xc/Xc.shape[0]+lam*torch.eye(D,device=DEV),
                             Xc.T@Yc/Xc.shape[0])
        maps[li]={'W':W,'bx':bx,'by':by}
    PR.LINS={}
    return maps

@torch.no_grad()
def main():
    t0=time.time()
    PR.LINS={}
    base=ce()
    assign={li:4 for li in LAYERS}
    maps=refit_maps(assign)
    install(assign,maps)
    cur=ce()-base
    print(f'start (all r4): +{cur:.3f} at {params_of(assign)/1e6:.2f}M',flush=True)
    history=[(params_of(assign),cur,dict(assign))]
    # greedy passes: try single-rank moves, joint-score the best candidates
    for sweep in range(3):
        improved=False
        # rank the candidate upgrades by approximate marginal (truncation delta
        # is unknown without eval -- use exact evals but only for upgrades of
        # the two largest-marginal layers per step, found by trying all cheaply
        # at one step size)
        cands=[]
        for li in LAYERS:
            r=assign[li]
            i=RANKS.index(r)
            for j in (i+1,i-1):
                if 0<=j<len(RANKS):
                    cands.append((li,RANKS[j]))
        best=None
        for li,r in cands:
            old=assign[li]; assign[li]=r
            install(assign,maps)
            c=ce()-base
            dp=params_of(assign)
            assign[li]=old
            # utility: improvement per extra param (or param saving per loss)
            dcost=c-cur; dpar=dp-params_of(assign)
            if dpar>0:
                util=-dcost/max(dpar,1)      # want most negative dcost per param
            else:
                util=(-dcost)*1e-9 + (0.002 if dcost<=0.005 else -1)  # free savings
            if best is None or util>best[0]:
                best=(util,li,r,c)
        util,li,r,c=best
        if c<cur-0.003 or (r<assign[li] and c<=cur+0.005):
            print(f'sweep {sweep}: L{li} -> r{r} (joint +{c:.3f})',flush=True)
            assign[li]=r
            maps=refit_maps(assign)   # refresh downstream fits
            install(assign,maps)
            cur=ce()-base
            history.append((params_of(assign),cur,dict(assign)))
            improved=True
        if not improved: break
    # budgets: continue upgrading greedily to fill 0.3M and 1M
    budgets=[0.1e6,0.3e6,1.0e6]
    pareto={}
    for B in budgets:
        while params_of(assign)<B:
            best=None
            for li in LAYERS:
                i=RANKS.index(assign[li])
                if i+1>=len(RANKS): continue
                r=RANKS[i+1]
                if params_of({**assign,li:r})>B: continue
                old=assign[li]; assign[li]=r
                install(assign,maps)
                c=ce()-base
                assign[li]=old
                if best is None or c<best[0]:
                    best=(c,li,r)
            if best is None: break
            c,li,r=best
            if c>=cur: break
            assign[li]=r
            cur=c
        maps=refit_maps(assign)
        install(assign,maps)
        cur=ce()-base
        pareto[B]=(params_of(assign),cur,dict(assign))
        print(f'budget {B/1e6:.1f}M: +{cur:.3f} at {params_of(assign)/1e6:.2f}M '
              f'ranks={[assign[li] for li in LAYERS]}',flush=True)
    PR.LINS={}
    p03=pareto[0.3e6][1]
    UNIFORM={0.1e6:1.807,0.3e6:1.660,1.0e6:1.541}  # section 159 refit curve
    pa=p03<=1.66-0.10
    pb=all(pareto[B][1]<UNIFORM[B] for B in budgets)
    ranks=[assign[li] for li in LAYERS]
    nz=[r for r in ranks if r>0]
    pc=(max(nz)/max(min(nz),1))>=4 if nz else False
    out={'base':base,'history':[(p,c) for p,c,_ in history],
         'pareto':{f'{B/1e6:.1f}M':{'params':pareto[B][0],'ce':pareto[B][1],
                                    'ranks':pareto[B][2]} for B in budgets},
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c':bool(pc)}
    print(f"\n(a) beats uniform-r16 by >=0.10 at 0.3M: {'HELD' if pa else 'FAILED'} (+{p03:.3f})")
    print(f"(b) dominates uniform curve at all budgets: {'HELD' if pb else 'FAILED'}")
    print(f"(c) allocation nonuniform (>=4x spread): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
