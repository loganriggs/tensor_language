"""Do heads share their content filters? Section 127: each score factor is a
~5-dim matched filter on-distribution. Sections 84/88: QK quadratic codes are
per-head. Test at the filter level: per head-factor, take the top-4 left/right
singular directions of K = C^{1/2} Wq^T Wk C^{1/2} (in the C-metric input
space); measure cross-head subspace alignment (median principal cosine) within
layers and across layers (0,2,5,9,13,16).

REGISTERED PREDICTIONS (skeptical, per the per-head QK result): (a) within-layer
median principal cos <= 0.35 (filters are private); (b) cross-layer <= 0.25;
null (c): random 4-dim subspaces in the top-64 C-metric ball give median cos
that both real numbers must exceed by <= 2x (i.e., real sharing is weak, near
the geometric floor). Alternative: within-layer >= 0.5 would mean a shared
attention lexicon and revise the per-head story."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
NH,HD,D=9,128,1152
LAYERS=(0,2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_filter_sharing_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    caps={li:[] for li in LAYERS}
    hs=[]
    for li in LAYERS:
        def mk(li=li):
            return lambda mod,inp: caps[li].append(
                F.rms_norm(inp[0].detach().reshape(-1,D).float(),(D,))) or None
        hs.append(m.transformer.h[li].attn.register_forward_pre_hook(mk()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    subs={}
    for li in LAYERS:
        X=torch.cat(caps[li]); Xc=X-X.mean(0)
        C=Xc.T@Xc/Xc.shape[0]
        ev,U=torch.linalg.eigh(C.double())
        Ch=((U*ev.clamp_min(0).sqrt())@U.T).float()
        a=m.transformer.h[li].attn
        for h in range(NH):
            for tag,(wq,wk) in (('s1',(a.c_q,a.c_k)),('s2',(a.c_q2,a.c_k2))):
                Wq=wq.weight.detach().float().view(NH,HD,D)[h]
                Wk=wk.weight.detach().float().view(NH,HD,D)[h]
                K=Ch@Wq.T@Wk@Ch
                Uk,S,Vk=torch.linalg.svd(K)
                subs[(li,h,tag)]=orth(torch.cat([Uk[:,:4],Vk[:4].T],dim=1))
    def med_cos(pairs):
        cs=[]
        for A,B in pairs:
            s=torch.linalg.svdvals(A.T@B)
            cs+= [float(x) for x in s[:4]]
        return sorted(cs)[len(cs)//2]
    g=torch.Generator(device=DEV).manual_seed(0)
    within=[];across=[]
    keys=list(subs)
    import itertools, random
    rng=random.Random(0)
    wpairs=[(subs[a],subs[b]) for a,b in itertools.combinations(keys,2)
            if a[0]==b[0] and a[1]!=b[1]]
    apairs=[(subs[a],subs[b]) for a,b in itertools.combinations(keys,2)
            if a[0]!=b[0]]
    rng.shuffle(wpairs); rng.shuffle(apairs)
    w=med_cos(wpairs[:300]); x=med_cos(apairs[:300])
    rpairs=[(orth(torch.randn(D,8,device=DEV,generator=g)),
             orth(torch.randn(D,8,device=DEV,generator=g))) for _ in range(100)]
    r=med_cos(rpairs)
    pa=w<=0.35; pb=x<=0.25; pc=(w<=2*r and x<=2*r)
    out={'within':w,'across':x,'random':r,
         'pred_a':bool(pa),'pred_b':bool(pb),'near_floor_c':bool(pc)}
    print(f'within-layer median cos {w:.2f} | cross-layer {x:.2f} | random {r:.2f}')
    print(f"(a) filters private within layer (<=0.35): {'HELD' if pa else 'FAILED'}")
    print(f"(b) cross-layer (<=0.25): {'HELD' if pb else 'FAILED'}")
    print(f"(c) near geometric floor (<=2x random): {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
