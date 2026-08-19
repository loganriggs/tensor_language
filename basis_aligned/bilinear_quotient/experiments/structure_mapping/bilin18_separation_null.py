"""Origin of the type separation (section 149): are the two watch-lists per
layer separate because training kept them apart, or by default? Same pipeline
with row-shuffled attention weights AND entry-shuffled MLP weights. REGISTERED
(skeptical): (a) shuffled separation is as clean as trained -- shuffled
attention-lexicon vs shuffled MLP-Gram alignment <= matched null + 0.1 at >= 4/5
layers (separation is the default); alternative: shuffled alignment HIGHER than
trained would mean training actively separates the types.

Prior context -- one watch-list per layer? Section 130: a layer's heads share a genuine filter
lexicon. Does it coincide with the layer's MLP input watch-list (top-8
eigendirections of the MLP's input-mode Lambda-Gram)? Per layer (2,5,9,13,16):
alignment between the attention lexicon (top-8 of the stacked head filters) and
the MLP Gram top-8, vs the covariance-matched null. REGISTERED: (a) median
alignment exceeds the matched null by >= 0.15 at a majority of layers (one
watch-list per layer); alternative: separate lexicons (echoes section 84
cross-type disjointness).

Prior context -- covariance-matched null for section 129. The 0.71 within-layer filter
alignment used an isotropic random floor (0.10), but the C-metric concentrates
every operator's singular vectors toward top-covariance directions. Fair null:
random subspaces drawn as Ch @ G (G gaussian), i.e. random directions weighted
by the same covariance. REGISTERED: (a) real within-layer alignment exceeds the
matched null by >= 0.2 (lexicon real); alternative: matched null >= 0.5 voids
the lexicon claim (sharing = covariance concentration).

Prior context -- do heads share their content filters? Section 127: each score factor is a
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
LAYERS=(2,5,9,13,16)
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_separation_null_results.json')

@torch.no_grad()
def main():
    t0=time.time()
    gsh=torch.Generator(device=DEV).manual_seed(9)
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
                Wq=Wq[:,torch.randperm(D,generator=gsh,device=DEV)]
                Wk=Wk[:,torch.randperm(D,generator=gsh,device=DEV)]
                K=Ch@Wq.T@Wk@Ch
                Uk,S,Vk=torch.linalg.svd(K)
                subs[(li,h,tag)]=orth(torch.cat([Uk[:,:4],Vk[:4].T],dim=1))
    # attention lexicon per layer: top-8 of stacked filter subspaces
    lex={}
    for li in LAYERS:
        mats=[subs[k] for k in subs if k[0]==li]
        Sfull=torch.cat(mats,dim=1)
        _,_,Vh=torch.linalg.svd(Sfull,full_matrices=False)
        U_,S_,_=torch.linalg.svd(Sfull@Sfull.T)
        lex[li]=orth(U_[:,:8])
    # MLP watch-list per layer: input-mode Gram top-8 (needs mlp INPUT stats)
    mlpin={li:[] for li in LAYERS}
    hs2=[]
    for li in LAYERS:
        def mk2(li=li):
            return lambda mod,inp: mlpin[li].append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs2.append(m.transformer.h[li].mlp.register_forward_pre_hook(mk2()))
    for i in range(0,24,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs2: h.remove()
    gram={}
    for li in LAYERS:
        X=torch.cat(mlpin[li]); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[li].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float()
        pf=torch.randperm(L.numel(),generator=gsh,device=DEV)
        L=L.reshape(-1)[pf].reshape(L.shape)
        pr=torch.randperm(R.numel(),generator=gsh,device=DEV)
        R=R.reshape(-1)[pr].reshape(R.shape)
        DD=Dw.T@Dw
        G=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        ev,U=torch.linalg.eigh(G.double())
        gram[li]=orth(U[:,ev.argsort(descending=True)[:8]].float())
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
    # covariance-matched null per layer: Ch @ gaussian, orthonormalized
    X2=torch.cat(caps[2]); Xc2=X2-X2.mean(0)
    C2=Xc2.T@Xc2/Xc2.shape[0]
    ev2,U2=torch.linalg.eigh(C2.double())
    Ch2=((U2*ev2.clamp_min(0).sqrt())@U2.T).float()
    mpairs=[(orth(Ch2@torch.randn(D,8,device=DEV,generator=g)),
             orth(Ch2@torch.randn(D,8,device=DEV,generator=g)))
            for _ in range(100)]
    rm=med_cos(mpairs)
    per={}
    for li in LAYERS:
        s_=torch.linalg.svdvals(lex[li].T@gram[li])
        per[li]=float(sorted(s_.tolist())[len(s_)//2])
        print(f'L{li:2d}: attention-lexicon vs MLP-watchlist median cos {per[li]:.2f}',
              flush=True)
    wins=sum(1 for li in LAYERS if per[li]<=rm+0.1)
    pa=wins>=4
    out={'per_layer':{str(k):v for k,v in per.items()},'matched_null':rm,
         'pred_a_one_watchlist':bool(pa),'n_above':wins}
    print(f'matched null {rm:.2f}')
    print(f"(a) shuffled separation <= null+0.1 at >=4/5 [see wins recount]: "
          f"{'HELD' if pa else 'FAILED'} ({wins}/5)")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
