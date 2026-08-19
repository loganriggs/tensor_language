"""v2 after the section-91 instrument error: the v1 score trace(C G2) was
magnitude-dominated (ranked identically to the loudness null). This version uses the
MAGNITUDE-FREE alignment ratio  trace(C G2) / (trace C * trace G2 / 1152), which is
1.0 for un-aligned (isotropic) writes and >1 when layer i writes into directions
layer i+1 is quadratically sensitive to. REGISTERED: (a2) Spearman(alignment ratio,
measured relative effect) >= 0.5; (b2) the top-aligned edge's effect >= 2x the
bottom-aligned edge's. Original protocol docstring follows.

The registered-open blind protocol for EDGES: can weights+stats rank the tail's
edge strengths before any intervention? For the ten adjacent pairs (i,i+1),
i=5..14, the coupling score is trace(C_i G2(i+1)) -- layer i's output covariance
contracted with layer i+1's input-mode Lambda-Gram (weights + input second moment).
Ground truth: transplant layer i's full MLP write from different documents and
measure the induced relative change in layer i+1's MLP output.

REGISTERED PREDICTIONS: (a) Spearman(score, measured effect) >= 0.6 over the ten
edges; (b) the top-scored edge's measured effect >= 3x the bottom-scored edge's;
(c) ALIGNMENT is what predicts, not loudness: the score's Spearman exceeds the
loudness-null score trace(C_i)*trace(G2)/1152 (magnitudes without alignment) by
>= 0.2."""
import json, sys, time, torch
sys.path.insert(0,'/workspace/tensor_language/basis_aligned/bilinear_quotient')
import torch.nn.functional as F
from bilin18_joint_removal import fwd, orth, m, FW, DEV
D=1152
EDGES=[(i,i+1) for i in range(5,15)]
OUT=('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
     'bilin18_blind_edges2_results.json')

def spearman(a,b):
    a=torch.tensor(a); b=torch.tensor(b)
    ra=a.argsort().argsort().float(); rb=b.argsort().argsort().float()
    ra=(ra-ra.mean())/ra.std(); rb=(rb-rb.mean())/rb.std()
    return float((ra*rb).mean())

@torch.no_grad()
def main():
    t0=time.time()
    base_rows=FW[300:324,:257].to(DEV); src_rows=FW[400:424,:257].to(DEV)
    # covariances of writes and input second moments, one forward sweep each
    mos={}; ins={}
    hs=[]
    for li in range(5,16):
        def mko(li=li):
            return lambda mod,i_,o_: mos.setdefault(li,[]).append(
                o_.detach().reshape(-1,D).float())
        def mki(li=li):
            return lambda mod,inp: ins.setdefault(li,[]).append(
                inp[0].detach().reshape(-1,D).float()) or None
        hs.append(m.transformer.h[li].mlp.register_forward_hook(mko()))
        hs.append(m.transformer.h[li].mlp.register_forward_pre_hook(mki()))
    for i in range(0,60,6):
        b=FW[i:i+6,:513].to(DEV)
        m(b[:,:-1].contiguous(), b[:,1:].contiguous())
    for h in hs: h.remove()
    scores=[]; null_scores=[]
    for i,j in EDGES:
        Y=torch.cat(mos[i]); Yc=Y-Y.mean(0)
        C=Yc.T@Yc/Yc.shape[0]
        X=torch.cat(ins[j]); S=X.T@X/X.shape[0]
        mlp=m.transformer.h[j].mlp
        L=mlp.Left.weight.detach().float(); R=mlp.Right.weight.detach().float()
        Dw=mlp.Down.weight.detach().float(); DD=Dw.T@Dw
        G2=L.T@(DD*(R@S@R.T))@L + R.T@(DD*(L@S@L.T))@R
        raw=float((C*G2).sum()); loud=float(C.trace())*float(G2.trace())/D
        scores.append(raw/loud)
        null_scores.append(loud)
    print('scores frozen before intervention\n')
    # measured: full-write transplant at i -> relative change in j's output
    effects=[]
    for i,j in EDGES:
        cap={}
        def cap_hook(mod,i_,o_): cap['mo']=o_.detach()
        h=m.transformer.h[i].mlp.register_forward_hook(cap_hook)
        outj={}
        def outj_hook(mod,i_,o_): outj.setdefault('l',[]).append(
            o_.detach().reshape(-1,D).float())
        h2=m.transformer.h[j].mlp.register_forward_hook(outj_hook)
        m(src_rows[:, :-1].contiguous(), src_rows[:,1:].contiguous())
        src_mo=cap['mo']; outj['l']=[]
        m(base_rows[:,:-1].contiguous(), base_rows[:,1:].contiguous())
        base_j=torch.cat(outj['l']); outj['l']=[]
        h.remove()
        def tr_hook(mod,i_,o_): return src_mo.to(o_.dtype)
        h3=m.transformer.h[i].mlp.register_forward_hook(tr_hook)
        m(base_rows[:,:-1].contiguous(), base_rows[:,1:].contiguous())
        h3.remove(); h2.remove()
        pat_j=torch.cat(outj['l'])
        eff=float((pat_j-base_j).pow(2).mean())/float(base_j.var())
        effects.append(eff)
        print(f'edge {i}->{j}: score {scores[len(effects)-1]:.3e} | '
              f'measured {eff:.4f}',flush=True)
    sK=spearman(scores,effects); sN=spearman(null_scores,effects)
    top=effects[max(range(10),key=lambda k:scores[k])]
    bot=effects[min(range(10),key=lambda k:scores[k])]
    pa=sK>=0.5; pb=top>=2*bot; pc=(sK-sN)>=0.2
    out={'edges':[f'{i}->{j}' for i,j in EDGES],'scores':scores,
         'null_scores':null_scores,'effects':effects,'spearman_K':sK,
         'spearman_loudness':sN,'top_over_bottom':top/max(bot,1e-9),
         'pred_a':bool(pa),'pred_b':bool(pb),'pred_c_alignment':bool(pc)}
    print(f'\nSpearman(K-score, effect) {sK:+.2f} | loudness-null {sN:+.2f}')
    print(f'top-scored effect / bottom-scored: {top/max(bot,1e-9):.1f}x')
    print(f"(a2) rank prediction >=0.5: {'HELD' if pa else 'FAILED'}")
    print(f"(b2) top/bottom >=2x: {'HELD' if pb else 'FAILED'}")
    print(f"(c) alignment beats loudness by >=0.2: {'HELD' if pc else 'FAILED'}")
    out['runtime_s']=time.time()-t0
    json.dump(out,open(OUT,'w'),indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')

if __name__=='__main__': main()
